# llama.cpp agent instructions

Give your llama.cpp-starting agent this:

```text
Start llama.cpp normally, but always append both stdout and stderr to a stable log file so llama-tracker can monitor it independently.

Use this pattern:

mkdir -p /var/log/llama
./llama-server -lv 4 [my existing llama-server options] >> /var/log/llama/server.log 2>&1

Do not pipe output through llama-tracker and do not start llama-tracker yourself. Preserve the existing server command/options, add `-lv 4` if absent, and add only the `>> /var/log/llama/server.log 2>&1` redirection.

If llama.cpp is managed by systemd, configure its ExecStart command to append stdout and stderr to that same file, or use systemd’s StandardOutput=append:/var/log/llama/server.log and StandardError=append:/var/log/llama/server.log.
```

Then run the tracker separately and keep it alive:

```bash
scripts/run-monitor-only.sh --log-file /var/log/llama/server.log --host 127.0.0.1 --port 8765
```

Use a different log path and port per llama.cpp instance if you run more than one.

## Truncate the log

To empty the log without deleting it:

```bash
: > /var/log/llama/server.log
```

Or:

```bash
truncate -s 0 /var/log/llama/server.log
```

The always-on tracker detects truncation and continues following new lines. Do
not delete the file while llama.cpp is writing to it.
