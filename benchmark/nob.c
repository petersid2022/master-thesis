#define NOB_IMPLEMENTATION
#define NOB_STRIP_PREFIX
#define NOB_EXPERIMENTAL_DELETE_OLD
#define NOB_WARN_DEPRECATED
#define EMPTY {0}
#include "nob.h"

Cmd cmd = EMPTY;

int main(int argc, char **argv)
{
  NOB_GO_REBUILD_URSELF(argc, argv);

  char *cwd;
  bool success;

  if (!(cwd = getcwd(NULL, 0))) goto cleanup;

  String_Builder sb = EMPTY;

  sb_append_cstr(&sb, cwd);
  sb_append_cstr(&sb, "/../llama.cpp/build/llama.pc");
  sb_append_null(&sb);

  cmd_append(&cmd, "pkg-config");
  cmd_append(&cmd, "--cflags");
  cmd_append(&cmd, "--libs");
  cmd_append(&cmd, sb.items);
  if (!nob_cmd_run(&cmd, .stdout_path = "/tmp/flags")) goto cleanup;

  sb.count = 0;

  if (!read_entire_file("/tmp/flags", &sb)) goto cleanup;
  sb.count -= 2;
  sb_append_null(&sb);

  cmd_append(&cmd, "clang");
  cmd_append(&cmd, "-o", "main");
  cmd_append(&cmd, "-std=c11");
  cmd_append(&cmd, "-Wall");
  cmd_append(&cmd, "-Wextra");
  cmd_append(&cmd, "-pedantic");
  cmd_append(&cmd, "-fsanitize=address");
  cmd_append(&cmd, "-I../llama.cpp/include/");
  cmd_append(&cmd, "-I../llama.cpp/ggml/include/");
  cmd_append(&cmd, "-L../llama.cpp/build/bin/");
  cmd_append(&cmd, "-Wl,-rpath,../llama.cpp/build/bin/");

  for (char *p = strtok(sb.items, " "); p; p = strtok(NULL, " "))
  {
    cmd_append(&cmd, p);
  }

  cmd_append(&cmd, "main.c");

  if (!nob_cmd_run(&cmd)) goto cleanup;

  success = true;

cleanup:
  free(cwd);
  free(sb.items);

  return success ? 0 : 1;
}
