# Schema

This file is yours to edit. The prose describes the ontology for humans; the
fenced `yaml` block at the bottom is what MarkMem actually parses. Edit both
together — adding a page type never requires a code change.

## Page types

- **user** — stable attributes, preferences, long-term facts about a person.
- **session** — episodic summary of one interaction: what happened, decisions, outcomes.
- **concept** — an enduring idea, strategy, or pattern.
- **entity** — a specific real-world object (account, property, client, service).
- **project** — an ongoing effort tying together sessions, entities, and concepts.

## Decay classes

- **fast** — short-lived, session-level detail. Half-life ~14 days.
- **medium** — entities/projects that change over months. Half-life ~90 days.
- **slow** — structural knowledge. Half-life ~365 days, never auto-archived.

## Trust ceilings

Maximum confidence a claim may carry, by provenance. Imported and inferred
claims can never masquerade as confident user statements.

## Relationship vocabulary

`relates_to`, `supersedes`, `part_of`, `depends_on`, `contradicts`

---

```yaml
page_types:
  user:    {decay: slow}
  session: {decay: fast, retain_days: 365}
  concept: {decay: slow}
  entity:  {decay: medium}
  project: {decay: medium}

decay_rules:
  fast:   {half_life_days: 14,  archive_below_confidence: 0.2}
  medium: {half_life_days: 90,  archive_below_confidence: 0.2}
  slow:   {half_life_days: 365, archive_below_confidence: null}

trust_ceilings:
  user_stated: 1.0
  human_edited: 1.0
  tool_derived: 0.9
  agent_inferred: 0.8
  imported: 0.6

relationships: [relates_to, supersedes, part_of, depends_on, contradicts]
```
