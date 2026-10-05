// Navigation only: use upstream parsing, never infer proof validity here.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { book_nil, book_load, book_fam, term_lower, name_key } from '../.tools/bend/bend2/bend.ts';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const book = book_nil();
const seen = new Map<string, string | null>();
// Since Bend 2.0.32 a namespace is the file's path from the root file's
// directory. Load every file against the repository root, so one file keeps
// one namespace across the several entries inspected here.
const top = root.replaceAll('\\', '/') + '/';
const namespaceOf = (file: string) => path.posix.relative(top, file).replace(/\.bend$/, '');
async function load(file: string) {
  await book_load(book, file, namespaceOf(file), seen, undefined, top);
}
await load(path.join(root, 'scripts/check_official.bend').replaceAll('\\', '/'));
const rootFile = path
  .join(root, 'convolution/proofs/native_inference_law.bend')
  .replaceAll('\\', '/');
const rootNamespace = namespaceOf(rootFile);
await load(rootFile);
// Inspect every maintained proof, including independent matrix/API contracts.
for (const owner of ['lib', 'convolution']) {
  const folder = path.join(root, owner, 'proofs');
  for (const filename of fs.readdirSync(folder)) {
    if (!filename.endsWith('.bend')) continue;
    await load(path.join(folder, filename).replaceAll('\\', '/'));
  }
}

const modules = [...seen]
  .filter(([, ns]) => ns)
  .map(([file, ns]) => ({
    namespace: ns!,
    file: path.relative(root, file).replaceAll('\\', '/'),
  }))
  .sort((a, b) => b.namespace.length - a.namespace.length);

function references(term: unknown): string[] {
  const names = new Set<string>();
  const pending: any[] = [term];
  while (pending.length) {
    const node = pending.pop();
    if (!node || typeof node !== 'object') continue;
    // Bend 2.0.24 keeps Nat/String literals compact. Record their datatype
    // dependencies without expanding a successor or character chain.
    if (node.$ === 'Lit') {
      const constructors = typeof node.v === 'number' ? ['Zero', 'Succ'] : ['SNil', 'SCon', 'Chr'];
      for (const constructor of constructors) {
        if (book.ctrs[constructor]) names.add(book_fam(book, constructor));
      }
    }
    if (['Ref', 'ADT', 'Ctr', 'Mat'].includes(node.$)) {
      names.add(node.k);
      if (book.ctrs[node.k]) names.add(book_fam(book, node.k));
    }
    for (const [key, value] of Object.entries(node)) {
      if (key !== 's') pending.push(value);
    }
  }
  return [...names].filter((name) => name in book.tlds).map(name_key).sort();
}

// Upstream spans contain source text, not filenames. Resolve those texts to the
// loaded files so an externally filled frozen law points to its actual body.
const sources = new Map<string, string>();
for (const [filename] of seen) {
  const lines = fs.readFileSync(filename, 'utf8').split('\n');
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trim();
    if (/^import(?:\s|$)/.test(line)) lines[index] = '';
    else if (line && !line.startsWith('#')) break;
  }
  sources.set(lines.join('\n'), path.relative(root, filename).replaceAll('\\', '/'));
}
function bodyFiles(term: unknown): string[] {
  const files = new Set<string>();
  const pending: any[] = [term];
  while (pending.length) {
    const node = pending.pop();
    if (!node || typeof node !== 'object') continue;
    const filename = sources.get(node.s?.src);
    if (filename) files.add(filename);
    for (const [key, value] of Object.entries(node)) if (key !== 's') pending.push(value);
  }
  return [...files].sort();
}

const definitions = Object.fromEntries(
  Object.entries(book.tlds).map(([name, term]) => {
    // Since Bend 2.0.35 an internal key is namespace:name; the output keeps namespace.name.
    const owner = modules.find((module) => name.startsWith(module.namespace + ':'));
    const signature = references(term_lower(term.T));
    const bodyTerm = term.$ === 'Def' ? (term.e ?? (term.v && term_lower(term.v))) : null;
    const body =
      term.$ === 'Def'
        ? references(bodyTerm)
        : term.c.flatMap((ctor) => references(term_lower(ctor.T)));
    return [
      name_key(name),
      {
        file: owner?.file ?? 'Base',
        body_files: bodyFiles(bodyTerm),
        kind: term.$,
        signature,
        body: [...new Set(body)].sort(),
      },
    ];
  })
);
const rootName = rootNamespace + '.native_inference_matches_matrix';
if (!(rootName in definitions)) throw new Error('Exact root definition not loaded');
function closure(seeds: string[]): Set<string> {
  const reached = new Set<string>();
  const pending = [...seeds];
  while (pending.length) {
    const name = pending.pop()!;
    if (reached.has(name)) continue;
    reached.add(name);
    const term = definitions[name];
    pending.push(...term.signature, ...term.body);
  }
  return reached;
}
const reached = closure([rootName]);
// These interfaces have a present purpose outside the convolution root.
// Their reference closure is retained too. This is not a second acceptance gate.
const retainedInterfaces: [string, string[], string][] = [
  ['lib/proofs/storage_refinement.bend', ['pack_reflected'], 'Every native Array has the shared storage observation'],
  ['lib/proofs/write_program.bend', ['program_preserves_balance', 'program_preserves_capacity'], 'Element-independent write-program shape and capacity contract'],
  ['convolution/proofs/native_inference_law.bend', ['every_strategy_matches_matrix'], 'Both execution strategies satisfy the unchanged full specification'],
  [
    'lib/proofs/traversal_array_refinement.bend',
    [
      'copy_traversal_refines_storage',
      'fold_refines_ordered_model',
      'map_loop_refines_model',
    ],
    'Shared actual traversal and ordered fold contracts',
  ],
  [
    'lib/proofs/array_layout_bounds.bend',
    ['view_address_is_inside'],
    'Rank-independent bounded lookup and storage ownership',
  ],
  [
    'lib/proofs/paired_read_refinement.bend',
    ['zip_preserves_selection'],
    'Static product zip composition for arbitrary element representations',
  ],
  [
    'convolution/proofs/matrix_convolution_law.bend',
    ['convolution'],
    'Frozen matrix-to-convolution contract',
  ],
  [
    'convolution/proofs/reference_refinement.bend',
    ['validated'],
    'Frozen reference implementation contract',
  ],
  [
    'convolution/proofs/matrix_axis_partitions.bend',
    ['output_channel_tiles'],
    'Ordered matrix partition API for optimization',
  ],
  [
    'convolution/proofs/matrix_axis_lengths.bend',
    ['columns', 'rows', 'channels'],
    'Dimensions of the same matrix API',
  ],
  [
    'convolution/proofs/array_api_validation.bend',
    ['validation_matches_spec', 'array_inference_matches_matrix'],
    'Native Array validation and complete Direct/Packed results for arbitrary supplied storage',
  ],
  [
    'convolution/proofs/array_tree_refinement.bend',
    ['packed_layout', 'read_dense'],
    'Logical Array API exercised by retained array comparisons',
  ],
  [
    'convolution/proofs/guarded_read_refinement.bend',
    ['bounded'],
    'Guarded read used by scalar and array comparisons',
  ],
  [
    'lib/proofs/loop_projection.bend',
    ['iteration_result_and_work', 'counter_progress'],
    'Shared parametric iteration work; concrete step cost remains an obligation',
  ],
  [
    'convolution/proofs/packing_lane_reads.bend',
    ['prepared_pixel_matches_direct_read'],
    'Runtime cached pixel geometry optimization contract',
  ],
  [
    'convolution/proofs/concrete_support_guard.bend',
    ['packed_guard_matches_original_conditions'],
    'Guard factoring preserves original conditions',
  ],
];
const retained = retainedInterfaces.flatMap(([file, names, reason]) =>
  names.map((name) => {
    const module = modules.find((item) => item.file === file);
    const symbol = module?.namespace + '.' + name;
    if (!(symbol in definitions)) throw new Error(`Retained interface is missing: ${file}:${name}`);
    return { symbol, file, reason };
  })
);
const maintained = closure([rootName, ...retained.map((item) => item.symbol)]);
const report = {
  scope:
    'Conservative source-term references, including types and all branches. Not a minimal proof dependency set, runtime call graph or checker verdict. Reflexive reduction can depend on definitions in types. Unreachable declarations require human review before deletion.',
  root: rootName,
  retained_interfaces: retained,
  root_modules: [
    ...new Set(
      [...reached].flatMap((name) => [definitions[name].file, ...definitions[name].body_files])
    ),
  ]
    .filter((file) => !file.startsWith('.tools/'))
    .sort(),
  definitions: Object.fromEntries(
    Object.entries(definitions)
      .filter(([, item]) => item.file !== 'Base')
      .map(([name, item]) => [
        name,
        {
          ...item,
          reachable_from_root: reached.has(name),
          needed_by_retained_interface: maintained.has(name),
        },
      ])
  ),
};
const output = path.join(root, 'out/results/proof-audit/dependencies.json');
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(
  `Root reference closure: ${reached.size} declarations across ${report.root_modules.length} modules. Navigation report: ${path.relative(root, output)}`
);
