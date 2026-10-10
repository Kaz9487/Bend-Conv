// Observability only: print shape and exact F32 bits, no numerical computation.
Term test_header(Env e, Term *f, IoWork *w) {
  printf("CASE %u %u %u\n", (u32)f[0], (u32)f[1], (u32)f[2]);
  return term_pak(CID(Unit), 0);
}
Term test_value(Env e, Term *f, IoWork *w) {
  printf("%08x\n", (u32)f[0]);
  return term_pak(CID(Unit), 0);
}
Term test_end(Env e, Term *f, IoWork *w) {
  puts("END");
  return term_pak(CID(Unit), 0);
}
static void __attribute__((constructor)) test_register(void) {
  io_eff(CID(Test.header), test_header);
  io_eff(CID(Test.value), test_value);
  io_eff(CID(Test.end), test_end);
}
