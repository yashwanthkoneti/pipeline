Put your input CSV here.

The path comes from PA_INPUT_PATH in .env, which defaults to
data/audit_input.csv.

Expected columns (names configurable in .env, matched case-insensitively):

  transaction_id       unique per row
  conversation_id      groups rows into a conversation
  request_created_on   orders the turns within a conversation
  request              the user's turn
  response             the agent's turn
  document_chunks      a list of chunks, each "**__KEY__:**VALUE" fields

History for each row is built from the earlier turns of the same
conversation_id, ordered by request_created_on.
