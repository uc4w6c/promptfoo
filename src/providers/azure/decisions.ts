import { z } from 'zod';
import { extractProviderResponseAttributes, withGenAISpan } from '../../tracing/genaiTracer';
import { renderVarsInObject } from '../../util/render';
import { callDecisions, parseDecisionsInput, requestSchema } from '../decisions/client';
import { AzureGenericProvider } from './generic';

import type {
  CallApiContextParams,
  CallApiOptionsParams,
  ProviderResponse,
} from '../../types/index';
import type { DecisionsConfig } from '../decisions/client';
import type { AzureCompletionOptions, AzureProviderOptions } from './types';

interface AzureDecisionsOptions
  extends Pick<
      AzureCompletionOptions,
      | 'apiHost'
      | 'apiBaseUrl'
      | 'apiKey'
      | 'apiKeyEnvar'
      | 'headers'
      | 'azureClientId'
      | 'azureClientSecret'
      | 'azureTenantId'
      | 'azureAuthorityHost'
      | 'azureTokenScope'
    >,
    DecisionsConfig {}

export class AzureDecisionsProvider extends AzureGenericProvider<AzureDecisionsOptions> {
  get handlesOwnRetries(): boolean {
    return true;
  }

  constructor(deploymentName: string, options: AzureProviderOptions<AzureDecisionsOptions> = {}) {
    if (!deploymentName.trim()) {
      throw new Error(
        'Azure Decisions requires a deployment name: use azure:decisions:<deployment name>',
      );
    }
    super(deploymentName, options);
  }

  id(): string {
    return `azure:decisions:${this.deploymentName}`;
  }

  async callApi(
    prompt: string,
    context?: CallApiContextParams,
    options?: CallApiOptionsParams,
  ): Promise<ProviderResponse> {
    options?.abortSignal?.throwIfAborted();
    if (context?.isGrading) {
      return {
        error:
          'Azure Decisions currently supports direct evaluation only; grading assertions are not supported.',
      };
    }
    const config: AzureDecisionsOptions = {
      ...this.config,
      ...(context?.prompt?.config as Partial<AzureDecisionsOptions>),
    };
    let body: z.infer<typeof requestSchema>;
    try {
      body = requestSchema.parse({
        model: this.deploymentName,
        input: parseDecisionsInput(prompt),
        questions: renderVarsInObject(config.questions, context?.vars),
        safety_identifier: renderVarsInObject(config.safety_identifier, context?.vars),
      });
    } catch (error) {
      const detail =
        error instanceof z.ZodError
          ? error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join('; ')
          : String(error);
      return { error: `Invalid Azure Decisions request: ${detail}` };
    }
    const baseUrl = this.getApiBaseUrl();
    if (!baseUrl) {
      throw new Error(
        'Azure Decisions requires apiBaseUrl, apiHost, or AZURE_OPENAI_API_BASE_URL.',
      );
    }
    // The v1 endpoint takes the deployment in the body and needs no api-version query.
    const apiUrl = /\/openai\/v1$/i.test(baseUrl)
      ? baseUrl
      : /\/openai$/i.test(baseUrl)
        ? `${baseUrl}/v1`
        : `${baseUrl}/openai/v1`;
    await this.ensureInitialized();
    options?.abortSignal?.throwIfAborted();
    return withGenAISpan(
      {
        system: 'azure.ai.openai',
        operationName: 'chat',
        model: body.model,
        providerId: this.id(),
        evalId: context?.evaluationId,
        testIndex: context?.testIdx,
        promptLabel: context?.prompt?.label,
        traceparent: context?.traceparent,
        requestBody: typeof body.input === 'string' ? body.input : JSON.stringify(body.input),
      },
      () =>
        callDecisions({
          body,
          config,
          apiUrl,
          headers: Object.fromEntries(
            Object.entries({
              'Content-Type': 'application/json',
              ...this.authHeaders,
              ...config.headers,
            }).filter(([name]) => name.toLowerCase() !== 'x-promptfoo-silent'),
          ),
          context,
          options,
          bustCache: context?.bustCache ?? context?.debug,
          providerName: 'Azure Decisions',
          allowCloudAuth: false,
        }),
      extractProviderResponseAttributes,
    );
  }
}
