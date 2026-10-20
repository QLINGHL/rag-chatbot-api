#!/usr/bin/env bash
# Demo walkthrough. Start the server first:
#   uvicorn app.main:app --reload --port 8000
# Then run:  bash demo.sh
set -e
BASE="${BASE:-http://localhost:8000}"

echo "== Health =="
curl -s "$BASE/health" | python3 -m json.tool

echo
echo "== Upload the sample handbook for tenant 'acme' =="
curl -s -X POST "$BASE/tenants/acme/documents" \
  -F "file=@sample_docs/northwind_handbook.txt" | python3 -m json.tool

echo
echo "== Ask a question (answer is in the Refunds section) =="
curl -s -X POST "$BASE/tenants/acme/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do customers have to request a refund?"}' \
  | python3 -m json.tool

echo
echo "== Tenant isolation: a different tenant has no documents =="
curl -s -X POST "$BASE/tenants/other-tenant/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do customers have to request a refund?"}' \
  | python3 -m json.tool
