# OpenMAIC / Pathwisse Main App TTS Static Audit

Date: 2026-07-04

## Scope and source

This is a separate static audit of the public `THU-MAIC/OpenMAIC` main app repository, used here as the OpenMAIC / Pathwisse main app source because no OpenMAIC/Pathwisse checkout exists in this workspace. Direct `git clone` and zip download attempts from GitHub were blocked by the execution environment with `CONNECT tunnel failed, response 403`, so the audit used GitHub web/raw source views for targeted files and searches.

Search terms requested:

- `/api/generate/tts`
- `arrayBuffer()`
- `base64`
- `data:audio`
- `use-discussion-tts`
- `kokoro`
- `lemonade`
- `audio/speech`
- `KITTEN_TTS_BASE_URL`
- `KOKORO_TTS_BASE_URL`

## High-level conclusion

OpenMAIC currently has a server-side TTS route at `app/api/generate/tts/route.ts` that returns JSON containing full base64-encoded audio. The route delegates to `lib/audio/tts-providers.ts`, where provider adapters generally fetch or receive a full audio payload with `response.arrayBuffer()` before returning bytes to the route. The route then converts the complete `Uint8Array` to base64 with `Buffer.from(audio).toString('base64')` and responds with `{ audioId, base64, format }`.

Playback therefore cannot begin from the server route until the upstream provider has fully generated/downloaded the audio, the server has base64-encoded the entire audio buffer, and the browser has received and parsed the JSON response.

KittenTTS is not currently wired as a provider in the audited OpenMAIC files. The current local-default/keyless path is `lemonade-tts`, configured with `defaultBaseUrl: 'http://localhost:13305/v1'`, `defaultModelId: 'kokoro-v1'`, and `supportedFormats: ['wav']`. To make KittenTTS the default, it should be connected in the same provider registry/provider implementation/server-config path that currently owns Lemonade/Kokoro TTS.

## Frontend hook/player files

The exact discussion hook path could not be enumerated from a full clone because cloning was blocked, but OpenMAIC release metadata and route comments confirm discussion TTS exists and calls `/api/generate/tts` after scene generation. The targeted `use-discussion-tts` search should be repeated in a local OpenMAIC checkout to identify the precise hook file, likely under `components` or a client-side hook subtree.

Expected client responsibilities from the API shape:

- Call `POST /api/generate/tts` for each speech action.
- Wait for the JSON response containing `base64` and `format`.
- Build a playable audio URL, commonly a `data:audio/<format>;base64,...` URL or Blob URL.
- Queue/play the decoded full audio item.

## API route files

### `app/api/generate/tts/route.ts`

Findings:

- Defines the single TTS generation API: `POST /api/generate/tts`.
- Requires `text`, `audioId`, `ttsProviderId`, and `ttsVoice`.
- Rejects `browser-native-tts` because that provider is client-only.
- Enforces server-side provider disablement.
- Uses `isServerConfiguredProvider`, `resolveTTSApiKey`, `resolveTTSBaseUrl`, and `resolveTTSModel` to merge server/client provider configuration.
- Calls `generateTTS(config, text)`.
- Converts the full returned `audio` buffer to base64 with `Buffer.from(audio).toString('base64')`.
- Returns JSON via `apiSuccess({ audioId, base64, format })`.

This is the main full-buffer/base64 conversion point for the discussion TTS path.

## Provider config files

### `lib/audio/types.ts`

`TTSProviderId` includes built-ins such as OpenAI, Azure, GLM, Qwen, VoxCPM, MiniMax, Doubao, ElevenLabs, Lemonade, and browser-native TTS. No `kitten-tts` provider ID was observed in the audited source views.

### `lib/audio/constants.ts`

`TTS_PROVIDERS` is the client-safe registry for TTS provider metadata. It includes the currently relevant local/default provider:

- `lemonade-tts`
- `requiresApiKey: false`
- `defaultBaseUrl: 'http://localhost:13305/v1'`
- `models: [{ id: 'kokoro-v1', name: 'Kokoro v1' }]`
- `defaultModelId: 'kokoro-v1'`
- `supportedFormats: ['wav']`
- Kokoro-style voice IDs such as `af_heart`, `af_alloy`, and language-specific variants.

No `KITTEN_TTS_BASE_URL` or `KOKORO_TTS_BASE_URL` environment variable was observed in the targeted raw source views. Lemonade uses the provider-level base URL and server config resolver path instead.

### `lib/server/provider-config.ts`

The TTS route imports `isServerConfiguredProvider`, `isServerTTSProviderDisabled`, `resolveTTSApiKey`, `resolveTTSBaseUrl`, and `resolveTTSModel` from this file. This is where server-pinned/default provider behavior should be audited or changed in a local checkout.

## Provider implementation files

### `lib/audio/tts-providers.ts`

`generateTTS(config, text)` dispatches by `config.providerId`.

Observed relevant dispatch cases:

- `openai-tts`
- `azure-tts`
- `glm-tts`
- `qwen-tts`
- `voxcpm-tts`
- `minimax-tts`
- `doubao-tts`
- `elevenlabs-tts`
- `lemonade-tts`
- `browser-native-tts` throws a server-side error.

Important full-buffer points:

- OpenAI-compatible providers call `/audio/speech` and then `response.arrayBuffer()`.
- Lemonade calls `${baseUrl}/audio/speech`, requests `response_format: config.format || 'wav'`, and then calls `response.arrayBuffer()`.
- VoxCPM vLLM/Omni path uses `/audio/speech` and `stream: false`, then `response.arrayBuffer()`.
- Azure, GLM, ElevenLabs, and Qwen audio download paths also use `response.arrayBuffer()`.
- MiniMax returns JSON containing hex audio, then converts that full string to bytes.
- Doubao parses response chunks and decodes base64 chunks into a combined `Uint8Array` before returning.

Important base64/data URL points:

- The route-level base64 conversion is in `app/api/generate/tts/route.ts`.
- VoxCPM reference-audio support includes `getVoxCPMDataAudioUrl(...)`, which constructs `data:${mediaType};base64,${base64}` for sending reference audio to a backend.
- VoxCPM Python API support includes `base64ToBlob(...)` for reference-audio upload form data.
- Doubao provider decodes provider-returned base64 chunks internally before returning bytes.

## Fallback behavior

Observed fallback and error behavior:

- Missing request fields return a 400 API error.
- `browser-native-tts` is rejected by `/api/generate/tts` and must be handled client-side.
- Server-disabled providers return 403.
- Managed server-configured providers ignore client-supplied API key/base URL.
- Unmanaged client-supplied base URLs are SSRF-validated before use.
- Provider `requiresApiKey` is enforced before provider dispatch.
- `TTSRateLimitError` is mapped to a 429 API response.
- Other provider failures become `GENERATION_FAILED` 500 responses.
- Custom TTS providers are handled through the OpenAI-compatible path.
- Provider defaults are supplied by `resolveTTSModel`, `resolveTTSBaseUrl`, and `TTS_PROVIDERS` metadata.

## Does playback wait for full JSON/base64?

Yes for `/api/generate/tts` playback. The route returns only after:

1. The provider implementation completes generation/download.
2. The provider implementation has the full audio bytes in memory.
3. The API route base64-encodes the complete bytes.
4. The route sends a JSON response containing the full base64 string.
5. The frontend receives/parses the full JSON and constructs a playable source.

This means the route is not streaming audio to the browser and does not support progressive playback from this path.

## Where KittenTTS should be connected as default

Recommended integration points in OpenMAIC:

1. Add a new `kitten-tts` provider ID to `lib/audio/types.ts`.
2. Add `kitten-tts` metadata to `TTS_PROVIDERS` in `lib/audio/constants.ts`, including:
   - `requiresApiKey: false`
   - `defaultBaseUrl` sourced through server/client config, ideally defaulting to the local KittenTTS server URL
   - model IDs matching the KittenTTS server, if applicable
   - voices matching the KittenTTS server voice list
   - `supportedFormats` matching the server output, likely `wav` and/or `mp3`
3. Implement `generateKittenTTS(config, text)` in `lib/audio/tts-providers.ts`.
4. Add `case 'kitten-tts'` in `generateTTS()`.
5. Add server resolver support in `lib/server/provider-config.ts`, including an environment variable such as `KITTEN_TTS_BASE_URL` if the project wants env-based defaulting.
6. Set the app/server default TTS provider to `kitten-tts` wherever `ttsProviderId` defaults are initialized in settings/store/server config.
7. If low-latency playback is a goal, consider adding a binary streaming route instead of returning JSON base64, or return a Blob/audio response directly from `/api/generate/tts` for KittenTTS.

## Follow-up checklist for a full local OpenMAIC checkout

Run these commands in the OpenMAIC repository once available locally:

```bash
rg -n "/api/generate/tts|arrayBuffer\(\)|base64|data:audio|use-discussion-tts|kokoro|lemonade|audio/speech|KITTEN_TTS_BASE_URL|KOKORO_TTS_BASE_URL" -S .
rg -n "ttsProviderId|TTS_PROVIDERS|resolveTTSBaseUrl|resolveTTSModel|browser-native-tts" -S app components lib
rg -n "new Audio|Audio\(|createObjectURL|data:audio|base64" -S app components hooks lib
```
