import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const compiler = path.join(root, '.tools', 'bend', 'bend2');
const args = process.argv.slice(2);
// Change only file arguments. In particular, guide topics, version and runtime
// arguments such as --threads 4 are not filesystem paths.
const fileArguments = new Set();
if (
  args.length &&
  !args[0].startsWith('-') &&
  !['guide', 'base', 'version', 'update'].includes(args[0])
) {
  fileArguments.add(0);
  for (let i = 1; i < args.length - 1; i++) {
    if (args[i] === '-o') fileArguments.add(++i);
  }
}
// Upstream runs on Linux and macOS, and on Windows through WSL: it resolves
// imports from POSIX real paths. On Windows this launcher runs itself in WSL
// with the same arguments; BEND_WSL_DISTRO selects the distribution.
if (process.platform === 'win32') {
  const linuxPath = (file) => {
    const full = path.resolve(file);
    const inside = path.relative(root, full);
    return inside.startsWith('..') || path.isAbsolute(inside)
      ? '/mnt/' + full[0].toLowerCase() + full.slice(2).replaceAll('\\', '/')
      : inside.replaceAll('\\', '/');
  };
  const distribution = process.env.BEND_WSL_DISTRO ? ['-d', process.env.BEND_WSL_DISTRO] : [];
  const linux = spawnSync('wsl.exe', [...distribution, '--cd', root, '--exec', 'sh', '-c',
    '. scripts/linux_environment.sh && exec node scripts/bend_launcher.mjs "$@"', 'bend_launcher',
    ...args.map((arg, i) => (fileArguments.has(i) ? linuxPath(arg) : arg))], { stdio: 'inherit' });
  if (linux.error) console.error(linux.error.message);
  process.exit(linux.error ? 1 : linux.status ?? 1);
}
const localBun = process.platform === 'linux' && process.arch === 'x64'
  ? path.join(root, '.tools', 'bun', 'node_modules', '@oven', 'bun-linux-x64', 'bin', 'bun')
  : null;
const bun = localBun && fs.existsSync(localBun) ? localBun : 'bun';
const requestedEntry = fileArguments.has(0) ? path.resolve(args[0]) : null;
for (const i of fileArguments) args[i] = path.resolve(args[i]);
const result = spawnSync(bun, [path.join(compiler, 'main.ts'), ...args], {
  cwd: compiler,
  encoding: 'utf8',
  maxBuffer: 16 * 1024 * 1024,
});
if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);
if (result.error) console.error(result.error.message);
// Some upstream checker failures print Error: but return zero. Normalize the
// process result; proof validity still comes exclusively from the checker.
const failed =
  result.error ||
  result.status !== 0 ||
  /^Error:/m.test((result.stdout || '') + (result.stderr || ''));
if (!failed && requestedEntry) {
  for (let i = 0; i < args.length - 1; i++) {
    if (args[i] !== '-o' || !fs.existsSync(args[i + 1])) continue;
    const output = args[i + 1];
    if (path.extname(output).toLowerCase() === '.c') {
      const localPython = path.join(root, '.venv', 'Scripts', 'python.exe');
      const python = process.platform === 'win32'
        ? (fs.existsSync(localPython) ? localPython : 'python')
        : (process.env.BENCH_PYTHON || 'python3');
      const inspection = spawnSync(python,
        ['-B', path.join(root, 'scripts', 'check_generated_c.py'), '--any-program', output],
        { cwd: root, encoding: 'utf8' });
      if (inspection.stdout) process.stdout.write(inspection.stdout);
      if (inspection.stderr) process.stderr.write(inspection.stderr);
      if (inspection.error || inspection.status !== 0) {
        console.error(inspection.error?.message || 'Generated C inspection failed.');
        process.exit(inspection.status || 1);
      }
    }
    // The app sandbox and interactive user can own this same local checkout.
    // Trust only this explicit project tool directory for this read command.
    const repository = path.dirname(compiler);
    const revision = spawnSync(
      'git',
      ['-c', 'safe.directory=' + repository, '-C', repository, 'rev-parse', 'HEAD'],
      { encoding: 'utf8' }
    );
    if (revision.status !== 0) {
      console.error(
        revision.error?.message || revision.stderr || 'Could not identify the Bend compiler commit.'
      );
      process.exit(1);
    }
    fs.writeFileSync(
      output + '.build.json',
      JSON.stringify({ compiler_commit: revision.stdout.trim(), source: requestedEntry, output }, null, 2)
    );
  }
}
process.exit(failed ? result.status || 1 : 0);
