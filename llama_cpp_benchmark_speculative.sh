#!/bin/bash
set -e

SERVER_URL="http://127.0.0.1:1234"
PROMPT="Write a Python class called Record with 20 properties: id, name, email, phone, address, city, state, zip_code, country, age, salary, department, role, manager, status, created_at, updated_at, is_active, score, notes. For each property implement a getter and setter using exactly this pattern: def get_X(self): return self._X and def set_X(self, value): self._X = value"
MAX_TOKENS=512
SEED=42
RUNS=3

SPEC_TYPES=(
	"none"
	"ngram_cache"
	"ngram_simple"
	"ngram_map_k"
	"ngram_map_k4v"
	"ngram_mod"
	"ngram_cache,ngram_simple,ngram_map_k,ngram_map_k4v,ngram_mod"
)

LOG_DIR="$HOME/master-thesis/logs/speculative"
mkdir -p "$LOG_DIR"

SERVER_BIN="$HOME/master-thesis/llama.cpp/build/bin/llama-server"
MODEL="$HOME/models/Qwen3.5-9B-Q8_0.gguf"

run_server() {
    local spec_type="$1"
    local log_file="$2"
    if [ "$spec_type" = "none" ]; then
        "$SERVER_BIN" -np 1 --port 1234 -m "$MODEL" --spec-type none 2>"$log_file" &
    else
        "$SERVER_BIN" -np 1 --port 1234 -m "$MODEL" --spec-type "$spec_type" 2>"$log_file" &
    fi
    echo $!
}

wait_for_server() {
    for i in $(seq 1 30); do
        if curl -sf "$SERVER_URL/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: server did not start in time" >&2
    exit 1
}

send_request() {
    curl -sf "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json, sys
print(json.dumps({
    'messages': [{'role': 'user', 'content': '$PROMPT'}],
    'max_tokens': $MAX_TOKENS,
    'seed': $SEED,
    'temperature': 0
}))
")"
}

echo "=== Speculative Decoding Benchmark ==="
echo "Prompt: $PROMPT"
echo "max_tokens=$MAX_TOKENS, seed=$SEED, runs=$RUN"
echo ""

RESULTS=()

for i in "${!SPEC_TYPES[@]}"; do
    spec_type="${SPEC_TYPES[$i]}"

    echo "--- spec-type: $spec_type ---"

    total_toks_per_sec=0

    for run in $(seq 1 $RUNS); do
        ts=$(date +%s)
        log_file="$LOG_DIR/${ts}_${spec_type}_run${run}"

        server_pid=$(run_server "$spec_type" "$log_file")
        wait_for_server
        sleep 1  # let warmup finish

        send_request >/dev/null

        kill "$server_pid" 2>/dev/null
        wait "$server_pid" 2>/dev/null

        toks_per_sec=$(grep "eval time" "$log_file" | grep -o '[0-9.]*\s*tokens per second' | grep -o '[0-9.]*' | tail -1)
        acceptance=$(grep "draft acceptance rate" "$log_file" | grep -o '[0-9.]*' | head -1 || echo "N/A")

        echo "  run $run: ${toks_per_sec} tok/s, acceptance=${acceptance}"
        total_toks_per_sec=$(python3 -c "print($total_toks_per_sec + ${toks_per_sec:-0})")
    done

    avg=$(python3 -c "print(round($total_toks_per_sec / $RUNS, 2))")
    echo "  avg: ${avg} tok/s"
    RESULTS+=("$spec_type: ${avg} tok/s")
    echo ""
done

echo "=== Summary ==="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
