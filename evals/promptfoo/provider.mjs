import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const evalDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(evalDir, '..', '..');
const python = process.env.PROMPTFOO_PYTHON || 'E:\\miniconda3\\python.exe';

function runProvider(input) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(evalDir, 'provider.py')], {
      cwd: repoRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python provider exited ${code}: ${stderr.slice(-1000)}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Python provider returned invalid JSON: ${error}; stderr=${stderr.slice(-500)}`));
      }
    });
    child.stdin.end(JSON.stringify(input));
  });
}

export default class HikingChatV2Provider {
  constructor(options = {}) {
    this.providerId = options.id || 'hiking-chat-v2-live-asgi';
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context) {
    const result = await runProvider({
      prompt,
      vars: context?.vars || {},
    });
    return { output: result.output, metadata: result.metadata };
  }
}
