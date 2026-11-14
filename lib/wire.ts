// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ── Upstream wires the baseline lacks: Anthropic and Gemini. ─────────────────────────────
// The entry routes to the best free model whatever its wire, so it must speak all three
// upstream. These are pure functions of their arguments: no clock, no randomness.

/** Anthropic's finish reason as OpenAI's finish_reason. */
function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'content_filter';
  if (finish === 'STOP') return 'stop';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY' || finish === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * The system message becomes `system`; user/assistant turns are rewritten with tool calls
 * as tool_use blocks and tool results as tool_result blocks; tools and tool_choice are
 * rewritten; max_tokens, temperature, top_p and stop pass through. JSON mode has no
 * Anthropic carrier, so it is not set here (routing keeps json requests off this wire).
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') }],
      });
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // user
    const content: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') {
      if (c !== '') content.push({ type: 'text', text: c });
    } else if (Array.isArray(c)) {
      for (const part of c) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') content.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const m = /^data:([^;]+);base64,(.+)$/.exec(url);
          if (m !== null) content.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          else content.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }
  const out: Json = { model, messages, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024 };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicRespToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      calls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl_aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: textParts.join('\n') !== '' ? textParts.join('\n') : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) }, finish_reason: anthropicStopToOpenAI(body.stop_reason) }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** One OpenAI message as Gemini `contents` entries. */
function openAIMessagesToGemini(messages: unknown[]): Json[] {
  const out: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    if (role === 'system') continue;
    const geminiRole = role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') {
      if (c !== '') parts.push({ text: c });
    } else if (Array.isArray(c)) {
      for (const part of c) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const m = /^data:([^;]+);base64,(.+)$/.exec(part.image_url.url);
          if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
        }
      }
    }
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
    }
    if (role === 'tool') {
      const resp: Json = {};
      resp[typeof raw.tool_call_id === 'string' && raw.tool_call_id !== '' ? raw.tool_call_id : 'result'] = { output: typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '') };
      out.push({ role: 'user', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' && raw.tool_call_id !== '' ? raw.tool_call_id : 'result', response: resp } }] });
      continue;
    }
    if (parts.length > 0) out.push({ role: geminiRole, parts });
  }
  return out;
}

/** OpenAI tools as Gemini functionDeclarations. */
function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const decls = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return decls;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request naming `model`. */
export function openAIToGeminiReq(body: Json, model: string): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const system = source.filter((m) => isRecord(m) && m.role === 'system').map((m) => textOf((m as Json).content)).join('\n');
  const out: Json = { contents: openAIMessagesToGemini(source) };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const decls = openAIToolsToGemini(body);
  if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function geminiRespToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const textParts: string[] = [];
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') textParts.push(part.text);
    else if (isRecord(part.functionCall)) {
      calls.push({ id: `call_${calls.length}`, type: 'function', function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) } });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl_gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: textParts.join('') !== '' ? textParts.join('') : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) }, finish_reason: geminiFinishToOpenAI(first.finishReason) }],
    usage: { prompt_tokens: numberOrZero(usage.promptTokenCount), completion_tokens: numberOrZero(usage.candidatesTokenCount), total_tokens: numberOrZero(usage.totalTokenCount) },
  };
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGeminiReq(body: Json, model: string): Json {
  const source = Array.isArray(body.messages) ? body.messages : [];
  const system = textOf(body.system);
  const contents: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') {
      if (c !== '') parts.push({ text: c });
    } else if (Array.isArray(c)) {
      for (const block of c) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          const resp: Json = {};
          resp[typeof block.tool_use_id === 'string' ? block.tool_use_id : 'result'] = { output: toolResultText(block.content) };
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : 'result', response: resp } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiRespToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const blocks: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string' && part.text !== '') blocks.push({ type: 'text', text: part.text });
    else if (isRecord(part.functionCall)) {
      blocks.push({ type: 'tool_use', id: `toolu_${blocks.length}`, name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', input: isRecord(part.functionCall.args) ? part.functionCall.args : {} });
    }
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: blocks,
    stop_reason: geminiFinishToAnthropic(first.finishReason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** One OpenAI SSE chunk line. */
export function openAIChunk(model: string, text: string, finish: string | null, usage: Json | null): string {
  const chunk: Json = { id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: text !== '' ? { content: text } : {}, finish_reason: finish }] };
  if (usage !== null) chunk.usage = usage;
  return `data: ${JSON.stringify(chunk)}\n\n`;
}

/**
 * A Gemini :streamGenerateContent SSE stream as OpenAI chat-completion chunks: each candidate
 * delta becomes a content chunk, then a final chunk carrying the finish reason and usage.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let usage: Json | null = null;
  let finish: string | null = null;
  let done = false;
  const emit = (text: string): string => openAIChunk(model, text, null, null);
  try {
    for (;;) {
      const { value, done: d } = await reader.read();
      if (d) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const t = line.trim();
          if (!t.startsWith('data:')) continue;
          const payload = t.slice(5).trim();
          if (payload === '' || payload === '[DONE]') continue;
          let chunk: unknown;
          try { chunk = JSON.parse(payload); } catch { continue; }
          if (!isRecord(chunk)) continue;
          const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
          const first = isRecord(candidates[0]) ? candidates[0] : {};
          const content = isRecord(first.content) ? first.content : {};
          const parts = Array.isArray(content.parts) ? content.parts : [];
          for (const part of parts) {
            if (isRecord(part) && typeof part.text === 'string' && part.text !== '') yield emit(part.text);
          }
          if (first.finishReason !== undefined) finish = geminiFinishToOpenAI(first.finishReason);
          if (isRecord(chunk.usageMetadata)) {
            usage = { prompt_tokens: numberOrZero(chunk.usageMetadata.promptTokenCount), completion_tokens: numberOrZero(chunk.usageMetadata.candidatesTokenCount), total_tokens: numberOrZero(chunk.usageMetadata.totalTokenCount) };
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const line of buffer.split('\n')) {
        const t = line.trim();
        if (!t.startsWith('data:')) continue;
        const payload = t.slice(5).trim();
        if (payload === '' || payload === '[DONE]') continue;
        try {
          const chunk: unknown = JSON.parse(payload);
          if (!isRecord(chunk)) continue;
          const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
          const first = isRecord(candidates[0]) ? candidates[0] : {};
          const content = isRecord(first.content) ? first.content : {};
          const parts = Array.isArray(content.parts) ? content.parts : [];
          for (const part of parts) {
            if (isRecord(part) && typeof part.text === 'string' && part.text !== '') yield emit(part.text);
          }
          if (first.finishReason !== undefined) finish = geminiFinishToOpenAI(first.finishReason);
        } catch { /* ignore trailing partial */ }
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield openAIChunk(model, '', finish ?? 'stop', usage);
  yield 'data: [DONE]\n\n';
  void done;
}

/**
 * A Gemini :streamGenerateContent SSE stream as the Anthropic event sequence.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let textIndex = -1;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop = 'end_turn';
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } })];
  };
  for await (const text of (async function* (): AsyncGenerator<string> {
    const reader = upstream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      for (;;) {
        const { value, done: d } = await reader.read();
        if (d) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
        let at = buffer.indexOf('\n\n');
        while (at !== -1) {
          const frame = buffer.slice(0, at);
          buffer = buffer.slice(at + 2);
          for (const line of frame.split('\n')) {
            const t = line.trim();
            if (!t.startsWith('data:')) continue;
            const payload = t.slice(5).trim();
            if (payload === '' || payload === '[DONE]') continue;
            let chunk: unknown;
            try { chunk = JSON.parse(payload); } catch { continue; }
            if (!isRecord(chunk)) continue;
            const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
            const first = isRecord(candidates[0]) ? candidates[0] : {};
            const content = isRecord(first.content) ? first.content : {};
            const parts = Array.isArray(content.parts) ? content.parts : [];
            for (const part of parts) {
              if (isRecord(part) && typeof part.text === 'string' && part.text !== '') yield part.text;
            }
            if (first.finishReason !== undefined) stop = geminiFinishToAnthropic(first.finishReason);
            if (isRecord(chunk.usageMetadata)) {
              inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
              outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
            }
          }
          at = buffer.indexOf('\n\n');
        }
      }
    } finally {
      reader.releaseLock();
    }
  })()) {
    const out: string[] = [...start()];
    if (textIndex === -1) {
      textIndex = 0;
      out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    }
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
    for (const e of out) yield e;
  }
  const out: string[] = [...start()];
  if (textIndex !== -1) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  out.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }));
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  for (const e of out) yield e;
}

/**
 * An Anthropic SSE stream as OpenAI chat-completion chunks: text deltas and tool-call
 * input deltas become content and tool_calls chunks, then a final chunk with the stop
 * reason and usage.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let stop = 'stop';
  let inputTokens = 0;
  let outputTokens = 0;
  const toolArgs = new Map<number, string>();
  const toolIds = new Map<number, string>();
  const toolNames = new Map<number, string>();
  const emitText = (text: string): string => openAIChunk(model, text, null, null);
  const emitTool = (index: number): string => {
    const args = toolArgs.get(index) ?? '';
    const chunk: Json = {
      id: 'chatcmpl_stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model,
      choices: [{ index: 0, delta: { tool_calls: [{ index, id: toolIds.get(index) ?? '', type: 'function', function: { name: toolNames.get(index) ?? '', arguments: args } }] }, finish_reason: null }],
    };
    return `data: ${JSON.stringify(chunk)}\n\n`;
  };
  try {
    for (;;) {
      const { value, done: d } = await reader.read();
      if (d) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let eventName = '';
        let dataLine = '';
        for (const line of frame.split('\n')) {
          const t = line.trim();
          if (t.startsWith('event:')) eventName = t.slice(6).trim();
          else if (t.startsWith('data:')) dataLine = t.slice(5).trim();
        }
        if (dataLine === '') continue;
        let data: unknown;
        try { data = JSON.parse(dataLine); } catch { continue; }
        if (!isRecord(data)) continue;
        if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        else if (eventName === 'message_delta' && isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStopToOpenAI(data.delta.stop_reason);
        else if (eventName === 'message_delta' && isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        else if (eventName === 'content_block_delta' && isRecord(data.delta)) {
          const idx = typeof data.index === 'number' ? data.index : 0;
          if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') yield emitText(data.delta.text);
          else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
            toolArgs.set(idx, (toolArgs.get(idx) ?? '') + data.delta.partial_json);
            yield emitTool(idx);
          }
        } else if (eventName === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
          const idx = typeof data.index === 'number' ? data.index : 0;
          toolIds.set(idx, typeof data.content_block.id === 'string' ? data.content_block.id : '');
          toolNames.set(idx, typeof data.content_block.name === 'string' ? data.content_block.name : '');
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield openAIChunk(model, '', stop, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens });
  yield 'data: [DONE]\n\n';
}
