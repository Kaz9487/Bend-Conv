// File I/O and phase clocks only. Reorder, packing, convolution and gathering are Bend.
static u64 lab_ticks[4];
Term lab_reps(Env e, Term *f, IoWork *w) {
  const char *s = getenv("CONV_REPS");
  return s ? (u32)atoi(s) : 1;
}
Term lab_param(Env e, Term *f, IoWork *w) {
  FILE *p = fopen("config.bin", "rb");
  u32 n;
  if (!p || fseek(p, 4 * (u32)f[0], SEEK_SET) || fread(&n, 4, 1, p) != 1)
    exit(2);
  fclose(p);
  return n;
}
Term lab_load(Env e, Term *f, IoWork *w) {
  const char *names[] = {"input.bin", "weight.bin", "bias.bin"};
  u32 n = f[1], d = 0;
  while ((1ull << d) < n)
    d++;
  Term z = 0, a = blk_new(e, false, d, 0, 1, &z);
  FILE *p = fopen(names[(u32)f[0]], "rb");
  if (!p || fread(blk_ptr(e.mem, term_loc(a), 0), 4, n, p) != n || fgetc(p) != EOF)
    exit(2);
  fclose(p);
  return a;
}
Term lab_start(Env e, Term *f, IoWork *w) {
  lab_ticks[0] = io_tick();
  return term_pak(CID(Unit), 0);
}
Term lab_reordered(Env e, Term *f, IoWork *w) {
  lab_ticks[1] = io_tick();
  return f[0];
}
Term lab_computed(Env e, Term *f, IoWork *w) {
  lab_ticks[2] = io_tick();
  return f[0];
}
Term lab_finish(Env e, Term *f, IoWork *w) {
  lab_ticks[3] = io_tick();
  return f[0];
}
Term lab_save(Env e, Term *f, IoWork *w) {
  FILE *p = fopen("bend.bin", "wb");
  if (!p || fwrite(blk_ptr(e.mem, term_loc(f[0]), 0), 4, (u32)f[1], p) != (u32)f[1])
    exit(2);
  fclose(p);
  blk_free(e, f[0]);
#ifdef LAB_INSTRUMENT
  p = fopen("copies.json", "w");
  if (!p)
    exit(2);
  fprintf(p, "{\"clone_calls\":%llu,\"clone_bytes\":%llu}\n", (unsigned long long)lab_clone_calls,
          (unsigned long long)lab_clone_bytes);
  fclose(p);
#endif
  p = fopen("bend.json", "w");
  if (!p)
    exit(2);
  fprintf(p, "{\"reorder_ms\":%.9f,\"compute_ms\":%.9f,\"gather_ms\":%.9f,\"total_ms\":%.9f}\n",
          (lab_ticks[1] - lab_ticks[0]) / 1e6, (lab_ticks[2] - lab_ticks[1]) / 1e6,
          (lab_ticks[3] - lab_ticks[2]) / 1e6, (lab_ticks[3] - lab_ticks[0]) / 1e6);
  fclose(p);
  p = fopen("bend_samples.jsonl", "a");
  if (!p)
    exit(2);
  fprintf(p, "{\"reorder_ms\":%.9f,\"compute_ms\":%.9f,\"gather_ms\":%.9f,\"total_ms\":%.9f}\n",
          (lab_ticks[1] - lab_ticks[0]) / 1e6, (lab_ticks[2] - lab_ticks[1]) / 1e6,
          (lab_ticks[3] - lab_ticks[2]) / 1e6, (lab_ticks[3] - lab_ticks[0]) / 1e6);
  fclose(p);
  return term_pak(CID(Unit), 0);
}
static void __attribute__((constructor)) lab_register(void) {
  io_eff(CID(Lab.reps), lab_reps);
  io_eff(CID(Lab.param), lab_param);
  io_eff(CID(Lab.load), lab_load);
  io_eff(CID(Lab.start), lab_start);
  io_eff(CID(Lab.reordered), lab_reordered);
  io_eff(CID(Lab.computed), lab_computed);
  io_eff(CID(Lab.finish), lab_finish);
  io_eff(CID(Lab.save), lab_save);
}
