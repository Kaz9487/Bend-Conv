// Foreign effects ONLY for binary tensor I/O, ownership transfer/copies,
// diagnostic output, and timing. All inference arithmetic lives in Bend.
static Term tensor_slots[512];
static bool tensor_used[512];
static u64 tensor_started;
static double tensor_dump_seconds;
static u64 tensor_copy_calls, tensor_copy_bytes, tensor_take_calls;
static u32 tensor_sample_index;
Term tensor_dump_enabled_run(Env environment, Term *arguments, IoWork *work) {
  const char *dump = getenv("BEND_DUMP");
  return term_pak((!dump || strcmp(dump, "0") != 0) ? CID(True) : CID(False), 0);
}
// Explicit effect boundaries preserve per-operation diagnostics while tensor
// owners stay in Bend. These effects perform no inference or storage access.
Term tensor_operation_begin_run(Env environment, Term *arguments, IoWork *work) {
  return term_pak(CID(Unit), 0);
}
Term tensor_operation_end_run(Env environment, Term *arguments, IoWork *work) {
  return term_pak(CID(Unit), 0);
}
Term tensor_repetitions_run(Env environment, Term *arguments, IoWork *work) {
  const char *text = getenv("BEND_REPETITIONS");
  if (!text) return 1;
  char *end;
  unsigned long count = strtoul(text, &end, 10);
  if (!*text || *end || count < 1 || count > UINT32_MAX || *text == '-') {
    fprintf(stderr, "BEND_REPETITIONS must be a positive U32\n");
    exit(2);
  }
  return (Term)count;
}
static void tensor_check_id(u32 tensor_id) {
  if (tensor_id >= 512) {
    fprintf(stderr, "tensor id out of range\n");
    exit(2);
  }
}
static const char *tensor_outdir(void) {
  const char *directory = getenv("BEND_OUTDIR");
  if (!directory) {
    fprintf(stderr, "BEND_OUTDIR is required\n");
    exit(2);
  }
  return directory;
}
Term tensor_load_run(Env environment, Term *arguments, IoWork *work) {
  u64 text_length = 0;
  char *filename = io_cstr(environment, arguments[0], &text_length);
  u32 tensor_id = arguments[1], count = arguments[2];
  tensor_check_id(tensor_id);
  FILE *file = fopen(filename, "rb");
  if (!file) {
    perror(filename);
    exit(2);
  }
  u32 depth = 0;
  while ((1ull << depth) < count)
    depth++;
  Term zero = 0;
  Term storage = blk_new(environment, false, depth, 0, 1, &zero);
  if (fread((void *)blk_ptr(environment.mem, term_loc(storage), 0), sizeof(float), count, file) !=
          count ||
      fgetc(file) != EOF) {
    fprintf(stderr, "wrong tensor file size: %s\n", filename);
    exit(2);
  }
  fclose(file);
  free(filename);
  if (tensor_used[tensor_id])
    blk_free(environment, tensor_slots[tensor_id]);
  tensor_slots[tensor_id] = storage;
  tensor_used[tensor_id] = true;
  return term_pak(CID(Unit), 0);
}
static Term tensor_get_run(Env environment, Term *arguments, IoWork *work) {
  u32 tensor_id = arguments[0];
  tensor_check_id(tensor_id);
  if (!tensor_used[tensor_id]) {
    fprintf(stderr, "missing tensor %u\n", tensor_id);
    exit(2);
  }
  tensor_copy_calls++;
  tensor_copy_bytes += (1ull << blk_span(tensor_slots[tensor_id])) * 8;
  return blk_copy(environment, tensor_slots[tensor_id]);
}
// Last-use transfer removes the slot owner; the Bend caller owns the only buffer.
static Term tensor_take_run(Env environment, Term *arguments, IoWork *work) {
  u32 tensor_id = arguments[0];
  tensor_check_id(tensor_id);
  if (!tensor_used[tensor_id]) {
    fprintf(stderr, "missing tensor %u\n", tensor_id);
    exit(2);
  }
  Term value = tensor_slots[tensor_id];
  tensor_used[tensor_id] = false;
  tensor_slots[tensor_id] = 0;
  tensor_take_calls++;
  return value;
}

// Native I/O trust boundary: the pinned runtime represents an Array by one
// complete block and rejects unequal-depth ANode construction (WONTFIX #808).
// Buffer's flattened ABI is owner, logical length, depth, equality word. This
// reads block metadata once; it neither scans nor changes tensor elements.
// Since Bend 2.0.35 foreign C names only Base's constructors and those of the
// file that declares the effect. The program hands one library Buffer to
// Tensor.buffer_like once; its constructor id is read from that value.
#ifdef CID(Tensor.buffer_like)
static u32 tensor_buffer_id;
static bool tensor_buffer_known;
Term tensor_buffer_like_run(Env environment, Term *arguments, IoWork *work) {
  Term fields[4];
  Term buffer = arguments[0];
  u32 id = (u32)term_aux(buffer);
  u32 arity = cid_arity(id);
  // tensor_buffer writes and tensor_save_buffer reads the flattened Buffer
  // fields by hand; stop if the compiler's layout no longer has four words.
  if (arity != 4) {
    fprintf(stderr, "Buffer constructor layout changed; update tensor_buffer\n");
    exit(2);
  }
  spare_free(environment, cls_fit(arity), ctr_take(environment, buffer, arity, fields));
  blk_free(environment, fields[0]);
  tensor_buffer_id = id;
  tensor_buffer_known = true;
  return term_pak(CID(Unit), 0);
}
static Term tensor_buffer(Env environment, Term storage, Term length) {
  if (!tensor_buffer_known) {
    fprintf(stderr, "Tensor.buffer_like must run before a buffer is read\n");
    exit(2);
  }
  u64 fields = heap_alloc(environment, 2);
  environment.mem[fields] = storage;
  environment.mem[fields + 1] = length;
  environment.mem[fields + 2] = blk_cls(storage);
  environment.mem[fields + 3] = 0;
  return term_ctr(tensor_buffer_id, fields);
}
Term tensor_get_buffer_run(Env environment, Term *arguments, IoWork *work) {
  return tensor_buffer(environment, tensor_get_run(environment, arguments, work), arguments[1]);
}
Term tensor_take_buffer_run(Env environment, Term *arguments, IoWork *work) {
  return tensor_buffer(environment, tensor_take_run(environment, arguments, work), arguments[1]);
}
#endif
static Term tensor_save_run(Env environment, Term *arguments, IoWork *work) {
  u32 tensor_id = arguments[0], count = arguments[2];
  tensor_check_id(tensor_id);
  Term storage = arguments[1];
  u64 text_length = 0;
  char *label = io_cstr(environment, arguments[3], &text_length);
  if (count > (1ull << blk_cls(storage))) {
    fprintf(stderr, "tensor capacity exceeded\n");
    exit(2);
  }
  if (tensor_used[tensor_id])
    blk_free(environment, tensor_slots[tensor_id]);
  tensor_slots[tensor_id] = storage;
  tensor_used[tensor_id] = true;
  const char *dump = getenv("BEND_DUMP");
  if (text_length && (!dump || strcmp(dump, "0") != 0)) {
    u64 before = io_tick();
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.bin", tensor_outdir(), label);
    FILE *file = fopen(path, "wb");
    if (!file) {
      perror(path);
      exit(2);
    }
    if (fwrite((void *)blk_ptr(environment.mem, term_loc(storage), 0), sizeof(float), count,
               file) != count) {
      perror(path);
      exit(2);
    }
    fclose(file);
    tensor_dump_seconds += (io_tick() - before) / 1e9;
  }
  if (text_length && (!dump || strcmp(dump, "0") != 0)) {
    printf("%s\n", label);
    fflush(stdout);
  }
  free(label);
  return term_pak(CID(Unit), 0);
}
#ifdef CID(Tensor.save_buffer)
// A ready public tensor leaves Bend as its Buffer: owner and logical length.
Term tensor_save_buffer_run(Env environment, Term *arguments, IoWork *work) {
  Term fields[4];
  Term buffer = arguments[1];
  u32 arity = cid_arity((u32)term_aux(buffer));
  spare_free(environment, cls_fit(arity), ctr_take(environment, buffer, arity, fields));
  Term saved[4] = {arguments[0], fields[0], fields[1], arguments[2]};
  return tensor_save_run(environment, saved, work);
}
#endif
Term tensor_begin_run(Env environment, Term *arguments, IoWork *work) {
  tensor_copy_calls = 0;
  tensor_copy_bytes = 0;
  tensor_take_calls = 0;
  tensor_dump_seconds = 0;
  tensor_started = io_tick();
  return term_pak(CID(Unit), 0);
}
Term tensor_end_run(Env environment, Term *arguments, IoWork *work) {
  double elapsed = (io_tick() - tensor_started) / 1e9;
  char path[4096];
  snprintf(path, sizeof(path), "%s/native_run.json", tensor_outdir());
  FILE *file = fopen(path, "w");
  if (!file) {
    perror(path);
    exit(2);
  }
  fprintf(file,
          "{\"seconds\":%.9f,\"dump_seconds\":%.9f,\"compute_and_store_seconds\":%.9f,\"host_copy_"
          "calls\":%llu,\"host_copy_bytes\":%llu,\"host_take_calls\":%llu,\"backend\":\"Bend "
          "native C; see build metadata for platform and compiler\"}\n",
          elapsed, tensor_dump_seconds, elapsed - tensor_dump_seconds,
          (unsigned long long)tensor_copy_calls, (unsigned long long)tensor_copy_bytes,
          (unsigned long long)tensor_take_calls);
  fclose(file);
  printf("Native forward: %.6f s (dump %.6f s)\n", elapsed, tensor_dump_seconds);
  snprintf(path, sizeof(path), "%s/native_samples.jsonl", tensor_outdir());
  file = fopen(path, tensor_sample_index ? "a" : "w");
  if (!file) { perror(path); exit(2); }
  fprintf(file, "{\"sample\":%u,\"seconds\":%.9f,\"dump_seconds\":%.9f}\n",
          tensor_sample_index++, elapsed, tensor_dump_seconds);
  fclose(file);
  return term_pak(CID(Unit), 0);
}
static void __attribute__((constructor)) tensor_register(void) {
  io_eff(CID(Tensor.load), tensor_load_run, 0);
#ifdef CID(Tensor.buffer_like)
  io_eff(CID(Tensor.buffer_like), tensor_buffer_like_run, 0);
#endif
#ifdef CID(Tensor.get_buffer)
  io_eff(CID(Tensor.get_buffer), tensor_get_buffer_run, 0);
#endif
#ifdef CID(Tensor.take_buffer)
  io_eff(CID(Tensor.take_buffer), tensor_take_buffer_run, 0);
#endif
#ifdef CID(Tensor.save_buffer)
  io_eff(CID(Tensor.save_buffer), tensor_save_buffer_run, 0);
#endif
  io_eff(CID(Tensor.begin), tensor_begin_run, 0);
  io_eff(CID(Tensor.end), tensor_end_run, 0);
#ifdef CID(Tensor.operation_begin)
  io_eff(CID(Tensor.operation_begin), tensor_operation_begin_run, 0);
  io_eff(CID(Tensor.operation_end), tensor_operation_end_run, 0);
  io_eff(CID(Tensor.dump_enabled), tensor_dump_enabled_run, 0);
#endif
#ifdef CID(Tensor.repetitions)
  io_eff(CID(Tensor.repetitions), tensor_repetitions_run, 0);
#endif
}
