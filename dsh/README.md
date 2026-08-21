# DSH LLM bridge

`llm-bridge.js` is AC Foundation's canonical adapter between `ac-llm` and the
native DSH LLM service. It owns only authenticated local transport and event
normalization; DSH retains provider credentials, routing, retries, and
streaming.

Product repositories may carry a generated copy. Their CI must compare its
SHA-256 digest with the Foundation source revision recorded in the product
runtime lock.
