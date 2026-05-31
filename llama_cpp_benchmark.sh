#!/usr/bin/env bash
set -euo pipefail

SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"
MODEL="/home/petrside/models/gemma-4-31B-it-Q3_K_M.gguf"
URL="http://127.0.0.1:1234"
PORT=1234
PROMPT='Write a Python class called Record with 20 properties: id, name, email, phone, address, city, state, zip_code, country, age, salary, department, role, manager, status, created_at, updated_at, is_active, score, notes. For each property implement a getter and setter using exactly this pattern: def get_X(self): return self._X and def set_X(self, value): self._X = value'
MAX_TOKENS=1024
SEED=42
TEMP=0
RUNS=3
TIMEOUT=7200
LOAD_WAIT=7200
LOG_DIR="/tmp/benchmark-logs"

SPECS=(
	none
	ngram_cache
	ngram_simple
	ngram_map_k
	ngram_map_k4v
	ngram_mod
	"ngram_cache,ngram_simple,ngram_map_k,ngram_map_k4v,ngram_mod"
)

mkdir -p "$LOG_DIR"

free_port() {
	pkill -f "llama-server.*--port.*${PORT}" 2>/dev/null || true
	command -v fuser >/dev/null 2>&1 && fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
	sleep 1
	command -v fuser >/dev/null 2>&1 && fuser -k -KILL "${PORT}/tcp" 2>/dev/null || true
	sleep 1
}

json_body() {
	BENCH_PROMPT="$PROMPT" BENCH_MAX="$MAX_TOKENS" BENCH_SEED="$SEED" BENCH_TEMP="$TEMP" python3 - <<'PY'
import json, os
print(json.dumps({
    "messages": [{"role": "user", "content": os.environ["BENCH_PROMPT"]}],
    "max_tokens": int(os.environ["BENCH_MAX"]),
    "seed": int(os.environ["BENCH_SEED"]),
    "temperature": float(os.environ["BENCH_TEMP"]),
}))
PY
}

wait_ready() {
	local i
	for i in $(seq 1 "$LOAD_WAIT"); do
		if curl -fsS --connect-timeout 3 --max-time 25 "$URL/health" >/dev/null 2>&1; then
			return 0
		fi
		if [ $((i % 60)) -eq 0 ]; then
			echo "benchmark: model still loading (${i}s / ${LOAD_WAIT}s max)..." >&2
		fi
		sleep 1
	done
	return 1
}

post_chat() {
	local j
	for j in $(seq 1 200); do
		if curl -fsS --connect-timeout 30 --max-time "$TIMEOUT" \
			-H "Content-Type: application/json" \
			--data-binary "$BODY" \
			"$URL/v1/chat/completions" >/dev/null 2>&1; then
			return 0
		fi
		sleep 2
	done
	return 1
}

toks() {
	grep "eval time" "$1" | grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' | tail -1
}

draft_rate() {
	grep "draft acceptance rate" "$1" | grep -oE '[0-9.]+' | head -1 || true
}

BODY=$(json_body)

free_port
for spec in "${SPECS[@]}"; do
	sum=0
	for r in $(seq 1 "$RUNS"); do
		tag="${spec//,/_}"
		log="$LOG_DIR/$(date +%s)_${tag}_r${r}.log"
		free_port
		"$SERVER_BIN" -np 1 --port "$PORT" -m "$MODEL" --spec-type "$spec" >/dev/null 2>"$log" &
		pid=$!
		if ! wait_ready; then
			echo "benchmark: /health never became ready (waited ${LOAD_WAIT}s)" >&2
			kill "$pid" 2>/dev/null || true
			exit 1
		fi
		kill -0 "$pid" 2>/dev/null || {
			echo "benchmark: llama-server exited before benchmark (check log)" >&2
			tail -n 50 "$log" >&2
			exit 1
		}
		if ! post_chat; then
			echo "benchmark: /v1/chat/completions failed after retries" >&2
			curl -fS --connect-timeout 30 --max-time 120 \
				-H "Content-Type: application/json" \
				--data-binary "$BODY" \
				"$URL/v1/chat/completions" >&2 || true
			kill "$pid" 2>/dev/null || true
			exit 1
		fi
		kill -TERM "$pid" 2>/dev/null || true
		wait "$pid" 2>/dev/null || true
		free_port
		t=$(toks "$log")
		t=${t:-0}
		sum=$(python3 -c "print($sum + float('$t'))")
		a=$(draft_rate "$log")
		echo "$spec run $r: ${t} tok/s accept=${a:-n/a} log=$log"
	done
	avg=$(python3 -c "print(round($sum / $RUNS, 2))")
	echo "AVG $spec: $avg tok/s"
	echo
done
