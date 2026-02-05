# Brainstorming - Remaining Open Points

The original brainstorming content has been reviewed against the implemented repository.
Completed sections were removed; this file now tracks only unresolved items.

## Open points

1. **Structured Outputs hardening (OpenAI)**
   - Current implementation uses JSON output parsing, but does not enforce the full strict schema from the initial brainstorm (with detailed required fields and enums).

2. **Gemini JSON robustness fallback**
   - Current implementation parses direct JSON text/fenced JSON only.
   - The brainstorm included a stronger fallback strategy for messy JSON envelopes.

3. **Traceability enforcement rules**
   - Brainstorm specified stronger guarantees for transcript-grounded claims and explicit `unsicher` tagging.
   - Current prompts request this behavior, but code-level validation is still missing.

4. **Long-transcript scalability path**
   - Brainstorm proposed chunking + merge / two-pass sectioning for very large transcripts.
   - Current implementation uses single-pass generation.

5. **Formatting fidelity parity with brainstorm template**
   - Some detailed markdown rendering conventions from the brainstorm (especially richer Z3-SMP formatting/sections) are simplified in current scripts.

---

If these open points are implemented, this file should be deleted.
