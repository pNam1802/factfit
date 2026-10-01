# Architecture decisions

Short records of the decisions that shape factfit: the context, the options, what was chosen,
and what it costs. New decisions copy [0000-template.md](0000-template.md).

| # | Decision | Status |
| --- | --- | --- |
| [0001](0001-langgraph.md) | Use LangGraph to orchestrate the tailoring agent | accepted |
| [0002](0002-sqlite.md) | Store app data in SQLite through SQLModel | accepted |
| 0003 | Check grounding with code first, LLM judge second | planned (with ablation results) |
| 0004 | Next.js + shadcn/ui instead of Streamlit | planned (week 2) |
| [0005](0005-openai.md) | Use the OpenAI API with Structured Outputs behind our own LLMClient | accepted |
