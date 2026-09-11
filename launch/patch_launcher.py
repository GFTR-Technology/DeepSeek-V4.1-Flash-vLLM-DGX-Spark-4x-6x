# Adaptive verification requires CUDA graphs; vLLM raises ValueError at config time under --enforce-eager.
# Make it follow EAGER: off when EAGER=1, else SPEC_ADAPT (default true). Idempotent.
import os, shutil, hashlib
p = os.path.expanduser("~/dsv41-tp4.sh")
s = open(p).read()
old_arg = '\\"enable_adaptive_verification\\":true}"'
new_arg = '\\"enable_adaptive_verification\\":${SPEC_ADAPT}}"'
guard = 'if [ "$SPEC" = "dspark" ]; then'
line = 'if [ "$EAGER" = "1" ]; then SPEC_ADAPT=false; else SPEC_ADAPT="${SPEC_ADAPT:-true}"; fi  # adaptive verification needs CUDA graphs\n'
if "SPEC_ADAPT" in s:
    print(os.uname().nodename, "already patched", hashlib.md5(s.encode()).hexdigest()[:8]); raise SystemExit
assert s.count(old_arg) == 1, "adaptive arg anchor count != 1"
assert s.count(guard) == 1, "spec guard anchor count != 1"
shutil.copy(p, p + ".bak-pre-adaptfix")
s = s.replace(old_arg, new_arg, 1).replace(guard, line + guard, 1)
open(p, "w").write(s)
print(os.uname().nodename, "patched; new md5", hashlib.md5(s.encode()).hexdigest()[:8], "(was 72ade078)")
