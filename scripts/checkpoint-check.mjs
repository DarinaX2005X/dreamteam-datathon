import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const localPython = resolve(root, 'backend/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const python = process.env.NAVIDU_CHECK_PYTHON || (existsSync(localPython) ? localPython : 'python');
const lintPython = process.env.NAVIDU_LINT_PYTHON || python;

function run(command, args, cwd = root) {
  const result = spawnSync(command, args, { cwd, stdio: 'inherit', windowsHide: true });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}

const commands = {
  lint() {
    run(lintPython, ['-m', 'ruff', 'check', 'backend/navidu', 'backend/tests', 'scripts', 'deploy']);
  },
  domain() {
    run(python, ['-m', 'pytest', 'backend/tests', 'deploy/tests', 'eval/test_convert_labels.py', '-q']);
  },
  sources() {
    run(python, ['-X', 'utf8', 'research/norms/verify_sources.py', '--schema']);
  },
  web() {
    const tsc = require.resolve('typescript/bin/tsc');
    const vite = resolve(dirname(require.resolve('vite/package.json')), 'bin/vite.js');
    const vitest = resolve(dirname(require.resolve('vitest/package.json')), 'vitest.mjs');
    run(process.execPath, [tsc, '--project', 'contracts/ts/tsconfig.json', '--noEmit']);
    run(process.execPath, [tsc, '--noEmit'], resolve(root, 'landing'));
    run(process.execPath, [vite, 'build'], resolve(root, 'landing'));
    run(process.execPath, [tsc, '-p', 'tsconfig.build.json'], resolve(root, 'landing'));
    run(process.execPath, [tsc, '--noEmit'], resolve(root, 'web'));
    run(process.execPath, [vitest, 'run'], resolve(root, 'web'));
    run(process.execPath, [vite, 'build'], resolve(root, 'web'));
  },
};

const requested = process.argv.slice(2);
for (const name of requested.length ? requested : Object.keys(commands)) {
  if (!commands[name]) throw new Error(`Unknown check: ${name}`);
  commands[name]();
}
