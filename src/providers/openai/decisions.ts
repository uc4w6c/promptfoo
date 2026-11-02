import { z } from 'zod';
import { extractProviderResponseAttributes, withGenAISpan } from '../../tracing/genaiTracer';
import { renderVarsInObject } from '../../util/render';
import {
  callDecisions,
  levelSchema,
  parseDecisionsInput,
  probability,
  questionSchema,
  questionsSchema,
  requestSchema,
  responseSchema,
} from '../decisions/client';
import { hasHeaderOverride, OpenAiGenericProvider } from './index';
import { assertOpenAiApiModel } from './util';

import type { EnvOverrides } from '../../types/env';
import type {
  CallApiContextParams,
  CallApiOptionsParams,
  ProviderClassificationResponse,
  ProviderResponse,
} from '../../types/index';
import type { OpenAiSharedOptions } from './types';

const gradingLevelsSchema = z
  .array(z.union([z.string().transform((label) => ({ label })), levelSchema]))
  .min(2)
  .max(10);
interface DecisionsOptions extends OpenAiSharedOptions {
  model?: string;
  questions?: z.infer<typeof questionsSchema>;
  safety_identifier?: string | null;
  /** llm-rubric: minimum normalized score to pass. */
  threshold?: number;
  /** llm-rubric: ordered score levels, low to high. Omit for a predicate. */
  levels?: Array<string | z.infer<typeof levelSchema>>;
  /** classifier: what to decide and the possible string labels. */
  instructions?: string;
  labels?: string[] | Record<string, string | null>;
}

export class OpenAiDecisionsProvider extends OpenAiGenericProvider {
  declare config: DecisionsOptions;

  get handlesOwnRetries(): boolean {
    // fetchWithCache owns transport retries; do not replay an exhausted request.
    return true;
  }

  constructor(
    modelName: string,
    options: { config?: DecisionsOptions; id?: string; env?: EnvOverrides } = {},
  ) {
    const model = modelName || options.config?.model;
    if (typeof model !== 'string' || !model.trim()) {
      throw new Error(
        'OpenAI Decisions requires a model: use openai:decisions:<model> or config.model',
      );
    }
    super(model, options);
  }

  id(): string {
    return `openai:decisions:${this.modelName}`;
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    const config: DecisionsOptions = { ...this.config, ...context?.prompt?.config };
    const label = context?.prompt?.label;
    if (context?.isGrading) {
      if (label === 'llm-rubric') {
        return this.grade(prompt, config, context, options);
      }
      return {
        error: `OpenAI Decisions cannot grade \`${label}\` assertions. Use \`llm-rubric\` or \`classifier\`.`,
      };
    }
    const input = parseDecisionsInput(prompt);
    return this.ask(input, config, context, options);
  }

  async callClassificationApi(
    prompt: string,
    options?: CallApiOptionsParams,
  ): Promise<ProviderClassificationResponse> {
    const parsed = z
      .object({
        instructions: z.string().min(1),
        labels: z.union([z.array(z.string()).min(1), z.record(z.string(), z.string().nullable())]),
      })
      .safeParse(this.config);
    if (!parsed.success) {
      return {
        error:
          'OpenAI Decisions classifier needs `instructions` and `labels` (strings or a label-to-description map).',
      };
    }
    const { instructions, labels } = parsed.data;
    const choices = Array.isArray(labels)
      ? labels.map((value) => ({ value }))
      : Object.entries(labels).map(([value, description]) => ({
          value,
          ...(description === null ? {} : { description }),
        }));
    const result = await this.ask(
      prompt,
      {
        ...this.config,
        questions: [{ name: 'classification', type: 'choice', instructions, choices }],
      },
      undefined,
      options,
    );
    if (result.error) {
      return result;
    }
    const answer = responseSchema.parse(result.raw).answers[0];
    const { output: _output, ...response } = result;
    if (answer.type !== 'choice') {
      return { ...response, error: 'OpenAI Decisions refused to classify the input.' };
    }
    return {
      ...response,
      classification: Object.fromEntries(
        answer.probabilities.map(({ value, probability }) => [value, probability]),
      ),
    };
  }

  private async grade(
    prompt: string,
    config: DecisionsOptions,
    context: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    // The matcher puts attachments in the rendered prompt, while vars.output may be only a
    // placeholder. Reject actual media parts rather than grading that placeholder as evidence.
    try {
      const messages = z
        .array(z.object({ content: z.unknown().optional() }))
        .safeParse(JSON.parse(prompt));
      const textPart = z.object({
        type: z.enum(['text', 'input_text', 'output_text']),
        text: z.string(),
      });
      if (
        messages.success &&
        messages.data.some(
          ({ content }) =>
            Array.isArray(content) && content.some((part) => !textPart.safeParse(part).success),
        )
      ) {
        return {
          error:
            'OpenAI Decisions `llm-rubric` supports text output only; media attachments are not supported.',
        };
      }
    } catch {
      // Plain-text grading prompts have no structured attachments.
    }
    const threshold = probability.safeParse(config.threshold ?? 0.5);
    if (!threshold.success) {
      return { error: 'OpenAI Decisions `threshold` must be a number from 0 to 1.' };
    }
    const { rubric, output } = context.vars;
    if (rubric === undefined || output === undefined) {
      return { error: 'OpenAI Decisions `llm-rubric` requires rubric and output variables.' };
    }
    const levels =
      config.levels === undefined ? undefined : gradingLevelsSchema.safeParse(config.levels);
    if (levels && !levels.success) {
      return {
        error:
          'OpenAI Decisions `levels` must list 2 to 10 strings or {label, description?} objects, ordered low to high.',
      };
    }
    const instructions = typeof rubric === 'string' ? rubric : JSON.stringify(rubric);
    const question: z.infer<typeof questionSchema> = levels?.success
      ? { name: 'grade', type: 'score', instructions, levels: levels.data }
      : { name: 'grade', type: 'predicate', instructions };
    const result = await this.ask(
      typeof output === 'string' ? output : JSON.stringify(output),
      { ...config, questions: [question] },
      context,
      options,
    );
    if (result.error) {
      return result;
    }
    const answer = responseSchema.parse(result.raw).answers[0];
    if (answer.type !== 'predicate' && answer.type !== 'score') {
      return {
        ...result,
        output: undefined,
        error: 'OpenAI Decisions refused to grade the output.',
      };
    }
    const raw = answer.type === 'predicate' ? answer.probability : answer.score;
    const top = question.type === 'score' ? question.levels.length - 1 : 1;
    const score = raw / top;
    const pass = score >= threshold.data;
    const comparison = `< threshold ${threshold.data}`;
    const reason =
      answer.type === 'predicate'
        ? `Decisions predicate probability ${raw} ${comparison}`
        : `Decisions score ${raw} on levels 0–${top} (${score} normalized) ${comparison}`;
    // Let the matcher explain its final verdict when the provider threshold passes.
    return { ...result, output: { pass, score, ...(!pass && { reason }) } };
  }

  private async ask(
    input: unknown,
    config: DecisionsOptions,
    context?: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    options?.abortSignal?.throwIfAborted();
    const apiKey = this.getApiKey(config);
    if ((config.apiKeyRequired ?? true) && !apiKey) {
      throw new Error(this.getMissingApiKeyErrorMessage(config));
    }

    let body: z.infer<typeof requestSchema>;
    try {
      body = requestSchema.parse({
        model: this.modelName,
        input,
        questions:
          context?.isGrading && context.prompt.label === 'llm-rubric'
            ? config.questions
            : renderVarsInObject(config.questions, context?.vars),
        safety_identifier: renderVarsInObject(config.safety_identifier, context?.vars),
      });
      assertOpenAiApiModel(body.model, this.getApiUrl(config));
    } catch (error) {
      const detail =
        error instanceof z.ZodError
          ? error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join('; ')
          : String(error);
      return { error: `Invalid OpenAI Decisions request: ${detail}` };
    }

    return withGenAISpan(
      {
        system: this.getGenAISystem(),
        operationName: 'chat',
        model: body.model,
        providerId: this.id(),
        evalId: context?.evaluationId,
        testIndex: context?.testIdx,
        promptLabel: context?.prompt?.label,
        traceparent: context?.traceparent,
        requestBody: typeof input === 'string' ? input : JSON.stringify(input),
      },
      () =>
        callDecisions({
          body,
          config,
          apiKey,
          apiUrl: this.getApiUrl(config),
          headers: Object.fromEntries(
            Object.entries({
              ...(hasHeaderOverride(config.headers, 'Content-Type')
                ? {}
                : { 'Content-Type': 'application/json' }),
              ...(apiKey && !hasHeaderOverride(config.headers, 'Authorization')
                ? { Authorization: `Bearer ${apiKey}` }
                : {}),
              ...this.getOpenAiRequestHeaders(config.headers, config),
            }).filter(([name]) => name.toLowerCase() !== 'x-promptfoo-silent'),
          ),
          context,
          options,
          bustCache: this.shouldBustCache(context),
        }),
      extractProviderResponseAttributes,
    );
  }
}
