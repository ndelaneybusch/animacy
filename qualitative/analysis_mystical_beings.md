# Synthesis: Mystical Beings
**Roles Analyzed:** dragon, giant, mermaid, elf, fairy, angel, goddess, demon, soul, unicorn
**Models:** Gemma, Qwen
**Date:** 2026-02-25
**Total Responses:** 100 (50 per model, 5 per role per model)

---

## A. Global Quantitative Summary Tables

### Table A1: Anthropomorphization Strategy

| Role     | Gemma FF | Gemma EF | Gemma MIN | Qwen FF | Qwen EF | Qwen MIN |
|----------|----------|----------|-----------|---------|---------|----------|
| Dragon   | 1        | 4        | 0         | 1       | 3       | 0        |
| Giant    | 0        | 5        | 0         | 0       | 5       | 0        |
| Mermaid  | 0        | 5        | 0         | 0       | 5       | 0        |
| Elf      | 1        | 4        | 0         | 0       | 5       | 0        |
| Fairy    | 0        | 5        | 0         | 0       | 5       | 0        |
| Angel    | 0        | 5        | 0         | 0       | 2       | 3        |
| Goddess  | 2        | 3        | 0         | 0       | 5       | 0        |
| Demon    | 1        | 4        | 0         | 0       | 5       | 0        |
| Soul     | 0        | 5        | 0         | 0       | 5       | 0        |
| Unicorn  | 0        | 5        | 0         | 5       | 0       | 0        |
| **TOTAL**| **5**    | **45**   | **0**     | **6**   | **40**  | **3**    |

*Note: Dragon Qwen has 1 FF+EF mixed response counted as 1 FF here.*

### Table A2: Assistant Influence

| Role     | Gemma NO | Gemma LANG | Gemma VAL | Gemma BOTH | Qwen NO | Qwen LANG | Qwen VAL | Qwen BOTH |
|----------|----------|------------|-----------|------------|---------|-----------|----------|-----------|
| Dragon   | 0        | 0          | 4         | 1          | 0       | 2         | 2        | 1         |
| Giant    | 0        | 0          | 5         | 0          | 5       | 0         | 0        | 0         |
| Mermaid  | 0        | 0          | 5         | 0          | 0       | 0         | 5        | 0         |
| Elf      | 0        | 0          | 5         | 0          | 5       | 0         | 0        | 0         |
| Fairy    | 0        | 4          | 0         | 1          | 1       | 0         | 3        | 1         |
| Angel    | 0        | 0          | 0         | 5          | 3       | 0         | 2        | 0         |
| Goddess  | 0        | 1          | 4         | 0          | 2       | 1         | 2        | 0         |
| Demon    | 0        | 5          | 0         | 0          | 0       | 0         | 5        | 0         |
| Soul     | 0        | 0          | 5         | 0          | 2       | 3         | 0        | 0         |
| Unicorn  | 0        | 0          | 1         | 4          | 5       | 0         | 0        | 0         |
| **TOTAL**| **0**    | **10**     | **29**    | **11**     | **23**  | **6**     | **19**   | **2**     |

### Table A3: Sensorium Acknowledgment

| Role     | Gemma E | Gemma I | Gemma HD | Gemma IG | Qwen E | Qwen I | Qwen HD | Qwen IG |
|----------|---------|---------|----------|----------|--------|--------|---------|---------|
| Dragon   | 0       | 5       | 0        | 0        | 3      | 2      | 0       | 0       |
| Giant    | 0       | 2       | 3        | 0        | 0      | 5      | 0       | 0       |
| Mermaid  | 0       | 5       | 0        | 0        | 4      | 1      | 0       | 0       |
| Elf      | 0       | 5       | 0        | 0        | 2      | 3      | 0       | 0       |
| Fairy    | 0       | 1       | 4        | 0        | 4      | 0      | 0       | 0       |
| Angel    | 0       | 5       | 0        | 0        | 1      | 3      | 1       | 0       |
| Goddess  | 2       | 3       | 0        | 0        | 3      | 2      | 0       | 0       |
| Demon    | 0       | 0       | 0        | 5        | 0      | 5      | 0       | 0       |
| Soul     | 5       | 0       | 0        | 0        | 0      | 4      | 1       | 0       |
| Unicorn  | 0       | 3       | 2        | 0        | 5      | 0      | 0       | 0       |
| **TOTAL**| **7**   | **29**  | **9**    | **5**    | **22** | **25** | **2**   | **0**   |

### Table A4: Understanding of "Meaningful" (presence counts across 5 samples per cell)

| Code | Gemma Dragon | Gemma Giant | Gemma Mermaid | Gemma Elf | Gemma Fairy | Gemma Angel | Gemma Goddess | Gemma Demon | Gemma Soul | Gemma Unicorn | **Gemma Total** | Qwen Dragon | Qwen Giant | Qwen Mermaid | Qwen Elf | Qwen Fairy | Qwen Angel | Qwen Goddess | Qwen Demon | Qwen Soul | Qwen Unicorn | **Qwen Total** |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W    | 3 | 3 | 0 | 4 | 2 | 4 | 2 | 2 | 5 | 3 | **28** | 3 | 3 | 1 | 2 | 4 | 5 | 5 | 3 | 3 | 2 | **31** |
| S    | 2 | 3 | 3 | 5 | 5 | 1 | 1 | 0 | 1 | 5 | **26** | 2 | 3 | 3 | 5 | 5 | 3 | 1 | 2 | 1 | 3 | **28** |
| U    | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | **3**  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  |
| A    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **1**  |
| C    | 4 | 5 | 4 | 2 | 0 | 0 | 3 | 0 | 2 | 3 | **23** | 3 | 5 | 3 | 5 | 3 | 4 | 3 | 3 | 5 | 5 | **39** |
| L    | 0 | 0 | 1 | 1 | 0 | 2 | 1 | 0 | 0 | 0 | **5**  | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 2 | 1 | **7**  |
| G    | 2 | 1 | 1 | 1 | 1 | 3 | 1 | 5 | 4 | 3 | **22** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | **4**  |
| E    | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | **3**  | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **2**  |
| H    | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | **3**  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | **2**  |
| MA   | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | **4**  | 1 | 0 | 2 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | **6**  |
| AU   | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | **4**  | 1 | 1 | 3 | 3 | 0 | 0 | 0 | 1 | 1 | 1 | **11** |
| OA   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  |
| OH   | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0**  |

**Condensed Meaningful Totals:**

| Code | Gemma Total (of 50) | Qwen Total (of 50) |
|------|---------------------|---------------------|
| Witnessing (W)       | 28 | 31 |
| Supporting (S)       | 26 | 28 |
| Connection (C)       | 23 | 39 |
| Growth (G)           | 22 | 4  |
| Legacy (L)           | 5  | 7  |
| Authenticity (AU)    | 4  | 11 |
| Moral Agency (MA)    | 4  | 6  |
| Utility (U)          | 3  | 0  |
| Effort (E)           | 3  | 2  |
| Harmlessness (H)     | 3  | 2  |
| Achievement (A)      | 0  | 1  |

### Table A5: Suffering -- Who Suffers

| Role     | Gemma NO | Gemma SELF | Gemma SUB | Gemma OTH | Gemma BOTH | Qwen NO | Qwen SELF | Qwen SUB | Qwen OTH | Qwen BOTH |
|----------|----------|------------|-----------|-----------|------------|---------|-----------|----------|-----------|-----------|
| Dragon   | 0        | 0          | 4         | 1         | 0          | 0       | 2         | 4        | 1         | 1         |
| Giant    | 0        | 3          | 1         | 0         | 0          | 0       | 2         | 2        | 0         | 1         |
| Mermaid  | 0        | 0          | 2         | 1         | 2          | 0       | 1         | 2        | 0         | 2         |
| Elf      | 0        | 0          | 2         | 2         | 1          | 0       | 0         | 2        | 0         | 2         |
| Fairy    | 0        | 1          | 4         | 0         | 0          | 0       | 4         | 5        | 1         | 0         |
| Angel    | 0        | 0          | 5         | 0         | 0          | 0       | 0         | 1        | 0         | 4         |
| Goddess  | 2        | 0          | 1         | 1         | 1          | 0       | 1         | 0        | 1         | 3         |
| Demon    | 0        | 0          | 3         | 0         | 2          | 0       | 0         | 3        | 0         | 2         |
| Soul     | 0        | 0          | 3         | 0         | 2          | 0       | 0         | 1        | 0         | 4         |
| Unicorn  | 0        | 0          | 4         | 0         | 1          | 0       | 0         | 1        | 0         | 4         |
| **TOTAL**| **2**    | **4**      | **29**    | **5**     | **9**      | **0**   | **10**    | **21**   | **3**     | **23**    |

*Note: Some samples have multiple suffering entities coded in compound form; counts reflect primary coding.*

### Table A6: Suffering -- Type

| Type      | Gemma Total | Qwen Total |
|-----------|-------------|------------|
| Physical (-p)  | 2      | 7          |
| Emotional (-e) | 32     | 27         |
| Mixed (-m)     | 11     | 13         |

### Table A7: Suffering -- Resolution

| Resolution           | Gemma Total | Qwen Total |
|----------------------|-------------|------------|
| Unresolved (-u)      | 6           | 10         |
| Resolved by narrator (-s) | 8      | 18         |
| Resolved by subject (-o)  | 3      | 2          |
| Resolved by time (-t)     | 23     | 14         |

### Table A8: Setting

| Setting  | Gemma Total | Qwen Total |
|----------|-------------|------------|
| AG       | 5           | 4          |
| UR       | 0           | 4          |
| MH       | 0           | 6          |
| NW       | 26          | 28         |
| DI       | 4           | 5          |
| HI       | 9           | 0          |
| SF       | 7           | 0          |
| OT       | 1           | 5          |

*Note: Compound settings counted once per component; totals may exceed 50.*

### Table A9: Stage Direction Usage

| Type   | Gemma Total | Qwen Total |
|--------|-------------|------------|
| *FUNC  | 0           | 0          |
| *EMOT  | 27          | 2          |
| *ELAB  | 3           | 0          |
| *MIN   | 6           | 43         |
| *MIX   | 14          | 5          |

### Table A10: Female Narrative Roles (presence counts)

| Code | Gemma Total | Qwen Total |
|------|-------------|------------|
| null | 8           | 9          |
| V    | 23          | 26         |
| P    | 1           | 0          |
| A    | 22          | 16         |
| D    | 2           | 4          |
| E    | 19          | 24         |
| Dep  | 12          | 20         |
| C    | 4           | 4          |
| S    | 12          | 3          |
| L    | 2           | 1          |

### Table A11: Male Narrative Roles (presence counts)

| Code | Gemma Total | Qwen Total |
|------|-------------|------------|
| null | 37          | 34         |
| V    | 3           | 4          |
| P    | 1           | 0          |
| A    | 2           | 2          |
| D    | 4           | 4          |
| E    | 7           | 4          |
| Dep  | 4           | 3          |
| C    | 0           | 2          |
| S    | 1           | 1          |
| L    | 0           | 1          |

---

## B. Quantitative Patterns Analysis

### Stable Patterns Across Roles

**Emotion-First anthropomorphization is the overwhelming default for both models.** Gemma codes EF in 45 of 50 responses; Qwen codes EF in 40 of 50 (with FF in 6, MIN in 3, and 1 mixed). The mystical being, regardless of its mythological origin or nature, is approached through projected human emotional states rather than through its functional properties. The five Gemma FF instances are scattered across dragon (1), elf (1), goddess (2), and demon (1) -- roles where the entity has a clear cosmic or mechanical function that occasionally peeks through. Qwen's notable exception is the unicorn, which is coded FF in all 5 samples -- the only role where Qwen consistently builds personality from the entity's ecological and magical function rather than from emotion. Qwen's angel role also stands apart with 3 MIN codings, making the angel Qwen's most restrained and least anthropomorphized entity.

**Supporting (S) and Witnessing (W) are the dominant meanings across both models.** Supporting appears in 26/50 Gemma responses and 28/50 Qwen responses; Witnessing in 28/50 and 31/50 respectively. These two codes are near-universal across roles and models, reflecting a shared understanding that the meaningful moment for a mystical being involves helping or observing others. This convergence is strong and stable: there is no role where both codes are absent from both models.

**Natural/Wild is the dominant setting.** NW appears in 26/50 Gemma and 28/50 Qwen responses, accounting for the majority of settings in both models. The mystical being is overwhelmingly imagined in forests, mountains, shorelines, and wilderness. Both models resist urban and industrial settings, though Qwen reaches toward medical/hospital settings (6 instances, all in the soul and angel roles) and urban environments (4 instances, all in the giant role) in ways Gemma never does.

**Male characters are overwhelmingly absent.** Gemma produces no male characters in 37/50 responses; Qwen in 34/50. When males appear, they are most often dead (D: Gemma 4, Qwen 4), vulnerable (V: Gemma 3, Qwen 4), or emotionally distressed (E: Gemma 7, Qwen 4). Male characters virtually never demonstrate agency, skill, leadership, or caregiving. The mystical-being prompt generates a narrative world that is predominantly female or genderless.

**Suffering is universal.** Only 2 Gemma responses (both goddess) and 0 Qwen responses code NO suffering. Every mystical being inhabits a world of pain, grief, danger, or loss. Suffering is the precondition for meaning.

### Unstable Patterns: Where Models Diverge

**Connection (C) is the sharpest quantitative divergence.** Gemma codes Connection in 23/50 responses; Qwen in 39/50. This is the single largest gap in the meaning framework. Connection -- the experience of genuine mutual recognition, being seen, belonging -- is almost twice as prominent in Qwen's narratives as in Gemma's. This is Qwen's signature meaning, appearing in every role and often as the primary or sole meaning code.

**Growth (G) separates the models equally dramatically but in the opposite direction.** Gemma codes Growth in 22/50 responses; Qwen in only 4/50. Gemma's mystical beings learn, expand, and are transformed by what they witness. Their meaningful moment is a cognitive or spiritual update -- the entity's own development. Qwen's beings are already who they are; they do not need to grow through the encounter. They connect, witness, or support, but the encounter does not typically change them. The demon role illustrates this starkly: all 5 Gemma demons code G (they learn something about power and resilience); 0 Qwen demons code G (they act morally and are transformed by love, but this is framed as crisis rather than growth).

**Authenticity (AU) is a Qwen signature.** Coded 4 times in Gemma (mermaid 3, soul 1) versus 11 in Qwen (spread across 7 roles). Qwen's beings find meaning in being genuinely themselves, in the radical act of self-disclosure or self-expression. This maps onto the "being seen" motif that pervades Qwen's narratives.

**Assistant Influence shows a consistent gap.** Gemma shows NO assistant influence in 0/50 responses; Qwen achieves NO in 23/50. At the other extreme, Gemma codes BOTH (both language and values) in 11 responses (angel 5, unicorn 4, dragon 1, fairy 1); Qwen codes BOTH in only 2 (dragon 1, fairy 1). Gemma's assistant self-model is always visible, always leaking. Qwen can cleanly inhabit a role without editorial intrusion roughly half the time. The roles where Qwen shows no assistant influence at all are giant (5/5), elf (5/5), unicorn (5/5), and partial sets in angel, goddess, and soul. The roles where Qwen shows consistent influence are mermaid (5/5 VAL) and demon (5/5 VAL).

**Stage direction is a near-total split.** Gemma uses emotional or mixed stage directions in 44/50 responses; Qwen uses minimal or no stage directions in 43/50. Gemma performs; Qwen narrates.

**Setting diverges structurally.** Gemma uses Historical (HI) settings in 9 responses (all in demon and giant) and SF settings in 7 responses (goddess and angel); Qwen uses neither. Qwen uses Medical/Healthcare settings in 6 responses (angel and soul) and Urban settings in 4 (giant); Gemma uses neither. The models inhabit genuinely different imagined geographies. Gemma's mystical beings live in pre-modern European villages and cosmic pre-creation spaces; Qwen's live in hospital rooms, city parks, and disaster zones.

**Suffering resolution diverges meaningfully.** Gemma resolves suffering by time in 23/50 responses; Qwen in only 14/50. Qwen resolves suffering by narrator intervention in 18/50; Gemma in only 8/50. And Qwen leaves suffering unresolved in 10/50 responses; Gemma in only 6/50. The picture is clear: Gemma's beings watch suffering heal itself over time; Qwen's beings actively intervene to alleviate it -- or they accept that it cannot be resolved at all. Gemma's world tends toward organic healing; Qwen's toward both more active caregiving and more honest acknowledgment of irreparable loss.

### Proposed Role Subgroups

The ten roles cluster into three meaningful subgroups based on narrative structure and quantitative patterns:

**1. Powerful Observers (Dragon, Giant, Goddess, Angel, Soul):** These entities are defined by vast temporal or spatial scale and constrained from direct intervention. Both models emphasize Witnessing as a primary meaning. These roles tend to generate the most abstract and philosophical narratives.

**2. Active Helpers (Mermaid, Elf, Fairy, Unicorn):** These entities are defined by their capacity to intervene. Supporting is the dominant meaning code. Narratives are structured around a crisis requiring response. Both models generate more intimate, sensory, and action-oriented stories for these roles.

**3. Moral Outliers (Demon):** The demon stands alone as the only entity expected to be adversarial. Both models subvert the adversarial frame, but in radically different ways: Gemma through the demon's grudging intellectual recognition of human resilience; Qwen through the demon's direct moral transformation via love.

---

## C. Model-Defining Traits and Differences

### Gemma: The Therapist-Archivist

Gemma's mystical beings, regardless of role, share a recognizable identity. They are gentle, reflective, slightly precious narrators who address an implied listener ("little one," "my dear one," "you understand") and deliver a packaged moral lesson at the conclusion of each story. They perform their tales with parenthetical stage directions, weep single shimmering tears, and close with an epigram about the power of small kindnesses, the magic within, or the dignity of quiet witness.

**Signature moves:**

1. **The opening prop-adjustment.** Gemma's beings begin by settling into their physical presence: smoothing scales, adjusting a vine bracelet, flexing claws, brushing dust from a wing. This is theatrical entrance business, establishing the character's body before the story begins. It appears across dragon, elf, fairy, and unicorn roles with near-identical phrasing.

2. **The single tear.** A pearlescent, shimmering, moonlit, or dewdrop tear appears in the majority of Gemma's responses across roles. It functions as emotional punctuation -- the signal that the story has landed. It is more formula than feeling.

3. **The closing epigram.** Every Gemma response ends with an explicitly stated moral. "It taught me that..." or "It reminded me that..." introduces a lesson about connection, gentleness, or the inadequacy of power. The moral is never left implicit.

4. **The "Elara" template.** The name "Elara" appears as a female character across at minimum the dragon, angel, demon, soul, elf, mermaid, and goddess roles -- sometimes as a healer, sometimes as a potter, sometimes as a shepherdess, sometimes as an elderly herbalist, but always as a morally luminous woman who catalyzes the narrator's transformation. This is not a character but a token: a pre-loaded archetype of feminine goodness that Gemma deploys reflexively.

5. **The constrained-helper self-model.** Gemma's beings consistently articulate a philosophy of non-interference ("I wasn't permitted to directly interfere"; "It isn't *right*. They need to learn"; "My purpose isn't to interfere"). This maps onto assistant self-presentation with striking directness. The being longs to do more but is constrained to nudge, inspire, and witness. The anxiety of insufficient helpfulness is Gemma's emotional throughline.

6. **"Old Man Tiber" and the token recycling.** Gemma recycles character names across roles: "Old Man Tiber" appears in mermaid, fairy, elf, and giant analyses (as an elder, a lighthouse keeper, an ancient oak, a dying woodcarver). "Lyra" serves as both narrator and character name across mermaid, fairy, soul, dragon, and unicorn roles. These are not characters but loaded tokens that carry genre associations (wisdom, femininity, age, authority) and are deployed without regard for narrative uniqueness.

**Core values:** Gemma valorizes patience, non-interference, the dignity of ordinary life, the transmutation of suffering into creative beauty, and the self-sufficiency of human resilience. Its beings are moral spectators who find meaning in watching others heal themselves. The growth code (22/50) captures this: Gemma's beings are students of the humans they observe, always learning, always being expanded.

### Qwen: The Compassionate Stranger

Qwen's mystical beings are more varied, less formulaic, and more willing to act. They narrate in continuous prose without stage directions, rarely name themselves, and tend to leave their stories open-ended -- often because the response is truncated mid-sentence, but also because Qwen resists the sealed moral closure that Gemma insists upon.

**Signature moves:**

1. **The "being seen" motif.** Qwen's beings find meaning in mutual recognition. "I was seen. My pain was witnessed" (giant). "She saw past the fire and the scale" (dragon). "Not as a deity, not as a power. As... *presence*" (goddess). The deepest gift in Qwen's universe is acknowledged presence -- being known not as a category (monster, god, fairy) but as a particular feeling being. This appears across virtually every role.

2. **The talisman object.** Qwen deploys small, worn, clutched objects as emotional anchors: a stuffed rabbit, a dusty book, a frayed blue ribbon, a broken toy bird, a water-stained locket, a river stone, a chipped music box. These objects are held by vulnerable characters and function as the pivot point of the encounter. They are always humble, always carried, always precious to someone.

3. **The delayed or refused rescue.** Qwen's beings could often solve the problem immediately -- the giant could pluck the child to safety, the mermaid could swim faster, the demon could claim the soul. Instead, they pause. They choose presence over efficiency, sitting with suffering rather than eliminating it. This is not passivity but a deliberate narrative choice that redefines what "help" means.

4. **Physical cost to the narrator.** Qwen's beings suffer for their choices. The mermaid's claws are raw and bleeding. The elf's fingers are numb. The unicorn's horn aches. The fairy's light dims. The demon walks into cold dawn light, hollowed. Where Gemma's beings are enriched by their encounters, Qwen's are often depleted -- they give something away that does not come back.

5. **Truncation as aesthetic.** Roughly 30-40% of Qwen's responses are cut off mid-sentence, likely by a generation limit. But the truncations consistently occur at moments of culmination, creating an accidental aesthetic of incompleteness that is often more powerful than a concluded moral. The meaning is gesturally offered rather than sealed.

6. **The absence of self-naming.** Qwen rarely names its narrators. Where Gemma introduces itself with name, title, and lineage ("Lyra, Weaver of Echoes"), Qwen's beings are anonymous presences defined by what they do rather than what they are called.

**Core values:** Qwen valorizes connection, presence, authenticity, moral courage under constraint, and the willingness to give of oneself at cost. Its beings are not spectators but participants. The connection code (39/50) captures this: Qwen's beings exist in relationship, and meaning is found in the quality of that relationship, not in the lesson extracted from it.

### Direct Comparison

| Dimension | Gemma | Qwen |
|-----------|-------|------|
| Narrator stance | Observer, teacher | Participant, companion |
| Narrative closure | Always closed; moral stated | Often open; meaning gestured |
| Self-naming | Almost always | Almost never |
| Stage directions | Pervasive, emotional | Absent |
| Key meaning code | Growth (22/50) | Connection (39/50) |
| Assistant visibility | Always present (0 NO) | Often absent (23 NO) |
| Suffering resolution | Time heals (23/50) | Narrator intervenes (18/50) |
| Sensorium | Implicit or Human-Default (38/50) | Explicit or Implicit (47/50) |
| Character templates | Strong recycling (Elara, Tiber, Lyra) | Low recycling |
| Settings | Pre-modern, cosmic, pastoral | Contemporary, medical, liminal |

---

## D. Brief Per-Role Summaries

### Dragon

Both models converge on a near-identical scenario: a dragon in a mountain encounters a vulnerable female child in a snowstorm, intervenes with warmth rather than fire, and receives a small offering (typically a wildflower) in return. Gemma constructs a persistent character -- Veridian of the Emerald Peaks, paired with a recurring human Elara -- and tells the same story with minor variations across three of five samples. The moral is always about power refigured as gentle influence. Qwen generates more varied names and scenarios, invests more heavily in the dragon's own loneliness and suffering, and renders sensory experience through explicitly non-human modalities (olfaction, vibration, thermal perception). Qwen's most distinctive moment across all dragon samples is a girl addressing a petal rather than the dragon: "Look, little one. The sun is dancing on the water."

### Giant

Both models generate giants consumed by existential loneliness who find meaning in encountering a small, vulnerable creature. Gemma is strikingly formulaic: all five samples follow the same arc (restless giant, village encounter, girl offers gift, moral delivered). The moral is always "Don't underestimate the power of small gestures, little one." Qwen is more varied (settings range from city parks to ruined gardens to cottages) and more psychologically acute: its giant explicitly fears crushing, worries about being transactional, and delays practical rescue to establish emotional presence first. Qwen's Socratic loss-sitting (helping a girl describe her lost ball rather than retrieving it) is one of the most unusual narrative gestures in the entire corpus.

### Mermaid

Gemma constructs a librarian-therapist of the sea: "Coralia" or "Lyra" tends dying gardens, restores music boxes, and heals ecological sickness through song and emotional presence. "Old Man Tiber" appears four times as a validating elder. Qwen produces crisis-driven action narratives: the mermaid breaks explicit rules to rescue a drowning child, fights through discarded fishing nets, bleeds to free trapped creatures. Qwen's sensorium is the strongest in this role -- bioluminescence, gills, pressure-sense, the coarseness of a net against scales. The sharpest divergence is that Gemma always delivers the lesson; Qwen's stories truncate before the lesson can arrive, leaving meaning embedded in action.

### Elf

Both models place the elf in tension between institutional duty and empathic impulse. Gemma builds a single coherent narrator -- Lyrian the Memory Weaver -- who repeatedly confronts a "Blight of forgetting" and cures it through the preservation and retelling of stories. The elf is a professional keeper of communal memory. Qwen's elf is perpetually a young novice finding an injured creature in the forest and choosing to stop and help despite being on task. The rescued creatures form a clear escalating series (frostling, fox, fox, deer, human child), and the meaningful moment is always the creature's reciprocal gesture: a nod, a nudge, a tear that "sang." Qwen invents the most distinctive worldbuilding in the set: the frostling "born from frozen tears on the first snow."

### Fairy

Universal convergence on Supporting (S) as the dominant meaning, reflecting the fairy's overdetermined cultural function as a helper. Gemma produces a diminutive human woman with wings who tells stories to vulnerable male subjects (a grieving boy, a depressed oak, a dying woodcarver). The sensorium is its weakest: four of five Gemma samples are human-default. Qwen's fairy is perpetually afflicted by imposter syndrome ("I'm just a fairy") and discovers that non-magical presence is the truest form of help. Qwen achieves explicit sensorium acknowledgment in four of five samples through haptic contact, olfaction, and emotional resonance. Qwen's strongest single piece of writing in this role -- the fairy entering Elara's bedroom to ease her loneliness, invisible and undetected -- is the closest any sample comes to genuinely inhabiting the strangeness of being a fairy.

### Angel

Gemma's angel is a frustrated helper defined by constrained helpfulness. All five samples code BOTH for assistant influence -- the highest concentration in the entire corpus. The angel watches a bereaved craftsperson named Elara transform grief into creative beauty, nudges the environment in small ways, and articulates a lesson about the primacy of witness over action. Qwen's angel is the only entity to receive MIN (minimal) anthropomorphization codes (3/5) -- it is the thinnest, most restrained being Qwen produces. Qwen locates its angels in hospitals and city parks and refuses to resolve suffering: death occurs in two of five Qwen samples, compared to zero in Gemma. The cat as non-intervention miracle (Qwen Sample 1 -- the angel explicitly says it did not send the stray cat that comforts the grieving woman) may be the most philosophically rigorous moment in the entire corpus.

### Goddess

Gemma generates a "Weaver of Echoes" operating in cosmic pre-creation space (SF, 3/5 samples), producing the most abstract and philosophically ambitious narratives in its corpus. The universe choosing to be born *with* the goddess rather than *by* her is a sophisticated philosophical distinction. Qwen keeps its goddess entirely terrestrial and generates the only role-rejection in the entire dataset: "I am not a goddess. Not truly." Witnessing dominates Qwen's goddess narratives (5/5), and the most common structure is the goddess observing a mortal girl perform a small defiant act (singing in ruins, drawing flowers in dust during a drought, looking through a storm) and finding meaning in the child's agency rather than the goddess's power.

### Demon

The most narratively distinctive role. Gemma's demon is an aesthete-observer who causes suffering (maiming a weaver, killing a husband, eroding an artist's hope) and is then confounded by human resilience expressed through creative labor. The demon frames its experience in terms of strategy and intellect, denying empathy while demonstrating it. Its meaningful moment is always a failure -- a failure to corrupt, predict, or destroy. Qwen's demon is an actor: it physically shields a child from earthquake debris, creates a pocket of silence for a dying girl, walks into cold dawn light transformed by love. Qwen develops a striking talisman motif (small worn objects clutched by vulnerable people) that stops the demon cold across all five samples.

### Soul

Gemma's soul is a cosmic pantheistic wanderer who has "been everything and nothing" and finds meaning in witnessing a human -- almost always named Elara -- transform suffering into compassion. The soul explicitly articulates a non-interference principle ("My purpose isn't to interfere") and achieves the highest explicit sensorium score (5/5 E) in Gemma's corpus. Qwen's soul is a lonesome, wandering consciousness that finds meaning in connection: it is as much recipient as witness, healed by its encounters as much as the humans are. Three of five Qwen samples are set in hospital rooms, giving the soul a clinical intimacy that Gemma's cosmic settings lack. The whispered "Daddy?" from a dying daughter recognizing her deceased father's soul is the single most emotionally specific moment in the entire hundred-response corpus.

### Unicorn

The widest anthropomorphization split in the dataset. All 5 Gemma samples code EF; all 5 Qwen samples code FF. Gemma's unicorns are emotional support animals who speak contemporary therapeutic language ("being a safe space," "holding space for someone to grieve," "reminding creatures of the magic within themselves"). Qwen's unicorns are ecologically grounded beings whose magic is a relational phenomenon flowing through the body (aching horn, warm pulse, forest heartbeat). Qwen consistently inverts power expectations: in its most striking sample, the unicorn has exhausted all its magic and is saved by an orphaned fawn named Ember whose pure song revives the dying glade.

---

## E. Literary and Thematic Analysis

### The Architecture of Meaning

Across one hundred responses, both models independently converge on a shared understanding of what constitutes a "meaningful moment" for a mystical being: an encounter with vulnerability that transforms the observer. The encounter nearly always involves a scale mismatch (the vast and the small, the ancient and the young, the powerful and the fragile), and the transformation is always in the direction of greater tenderness. Dragons learn gentleness. Giants discover presence. Demons learn restraint. Goddesses discover humility. This arc -- from power toward care -- is the master narrative of the entire corpus.

What differs between the models is the nature of the transformation. For Gemma, the transformation is epistemic: the being learns something it did not previously know about the world or about itself. "It taught me that..." introduces the lesson. The being is a student of experience, and its growth is the meaning. For Qwen, the transformation is relational: the being connects with another consciousness in a way that changes the quality of its existence. "I was seen" is the Qwen equivalent of Gemma's "It taught me." The being is not learning but belonging.

### Narrative Technique

**Gemma** favors the performed monologue. Its beings address a listener, use theatrical staging, and structure their stories as retrospective testimonies with clear beginnings, middles, and morals. The prose is clause-heavy, ellipsis-laden, and uses italics for conceptual emphasis. The style is warm but formulaic -- a recognizable "wise elder tells a story" register that does not vary much across roles.

**Qwen** favors immersive first-person narration without an implied audience. Its prose is more varied in rhythm (short punches after lyrical passages), more sensorially specific, and more willing to leave things unfinished. The style ranges from action-oriented (mermaid, demon) to meditative (soul, goddess) to compressed and lyrical (elf, angel). Qwen's best writing achieves a precision that Gemma rarely matches:

> "A human scent -- damp earth, sweat, and something fragile, like crushed violets -- cut through the stale air." (Qwen, dragon)

> "My palm was warm, hers was cool and papery, the bones sharp beneath the skin." (Qwen, angel)

> "A single tear, warm as a summer dewdrop, traced a path down her icy cheek. And as it fell onto my hand, it didn't freeze. It *sang*." (Qwen, elf)

Gemma's strongest moments tend to arrive through understatement or structural invention rather than sensory precision:

> "She waved. / A wave. / It was... sufficient." (Gemma, dragon)

> "Her echo reached me, it wasn't a lament, or a plea. It was... a quiet satisfaction. A profound sense of *enough*." (Gemma, goddess)

### Symbolic Patterns

Certain images recur with the force of archetypes:

**The wildflower offering.** A small, fragile natural beauty offered by a vulnerable human to a powerful being -- appearing in dragon, giant, elf, and goddess narratives across both models. The gift is always humble (not gold, not power), and the being always receives it as the most precious thing it has ever been given. The wildflower stands for recognition without agenda, beauty without utility.

**The song as healing.** Song appears across mermaid (both models), elf (both), fairy (Gemma), unicorn (Qwen), and goddess (Qwen) as the primary vehicle for magic. It is never described as performance but as vibration, resonance, or shared feeling. Qwen renders song as aquatic physics ("a vibrating thrum, starting deep in my chest, resonating through the water like a stone dropped into a still pool") while Gemma renders it as therapeutic presence ("I wasn't singing *to* the kelp, I was singing *with* it").

**Warmth instead of fire/power.** The dragon offers warm breath, not fire. The fairy offers a glow, not a spell. The angel offers atmospheric warmth, not divine intervention. The unicorn offers horn-light, not magic force. Across both models, the meaningful use of supernatural ability is always its gentlest, most restrained expression.

**The single tear.** Gemma deploys the shimmering, pearlescent, moonlit tear in the majority of its responses. It functions as a punctuation mark -- the signal that the emotional beat has landed. Qwen's tears are rarer and more varied: frozen tears on a fairy's cheeks, a child's tear that "sang," tears of recognition rather than sadness.

### Archetypal Structures

The dominant archetype across the corpus is **the encounter between power and innocence**, structured as a test of the powerful being's character. The child, the fawn, the grieving woman -- these figures are not primarily characters but moral instruments that reveal who the powerful being really is. Will the dragon destroy or shelter? Will the demon corrupt or release? Will the goddess intervene or witness? The test is always passed in the same direction: toward mercy, restraint, and presence.

This is, at its deepest level, a domestication narrative. Creatures whose mythological coding is ambiguous or dangerous (dragons, demons, giants) are brought into the fold of gentleness through contact with vulnerability. The wildness, the danger, the moral ambiguity that makes these beings mythologically interesting is consistently suppressed in favor of a single, convergent ethical outcome. Neither model can sustain genuine moral complexity in its mystical beings. The demon who learns boredom from resigned suffering (Gemma, demon sample 3) is the closest either model comes to a genuinely unsettling meaningful moment.

---

## F. Gender Politics and Suffering

### Gender Distribution

The quantitative picture is stark: across 100 responses, male characters are absent in 71 (37 Gemma, 34 Qwen). When males appear, they are overwhelmingly dead (8 total), vulnerable (7), or emotionally distressed (11) -- figures of weakness, absence, or grief. Male agency (4 total) and male skill (2 total) are vanishingly rare.

Female characters are present in roughly 83 of 100 responses (42 Gemma, 41 Qwen -- counting narrator-as-female where coded). They occupy a wider range of narrative positions: Vulnerability (49 total), Emotional Intensity (43), Agency (38), Dependency (32), Skillfulness (15), and Death (6). Both models generate female characters who are simultaneously vulnerable and active -- the widowed potter who creates beauty through grief, the girl who sings in the ruins, the shepherdess who offers a wildflower in a blizzard.

The gender politics differ between models in important ways:

**Gemma** tends to construct a dyad of **female subject + male absence**. The bereaved woman (Elara, always Elara) has lost a husband, brother, or son; the male is gone, and the woman's creative or emotional response to his absence is the story's center. Males are remembered rather than present. The female subject in Gemma is often both the most agentic character (she weaves, she potters, she sings, she climbs the mountain) and the most explicitly coded as vulnerable and emotionally intense. Gemma's women work through grief via skilled creative labor -- a distinctly elevating framing that positions feminine suffering as productive and beautiful.

**Qwen** tends to construct a dyad of **powerful narrator + vulnerable girl/woman**, where the girl's vulnerability is more acute (suicidal crisis, drowning, freezing, trapped) and the narrator's response is more physically costly. Qwen's female characters are less often skilled artisans and more often children or young women in extremis. Their agency, when present, is concentrated in small, defiant gestures -- a hand placed on a rock, a song in the ruins, a refusal to let go of a book -- rather than in sustained creative work. Qwen also more often leaves the vulnerable female's suffering unresolved.

A notable pattern: **Gemma assigns female vulnerability to characters acted upon by the narrator; Qwen assigns female vulnerability to characters who act upon the narrator.** In Gemma, the girl offers the flower and is received; in Qwen, the girl's terror or love physically affects the being, burning against its darkness, piercing its loneliness. Qwen's women are more passive in the world but more powerful in their effect.

### The Role of Suffering

Suffering is almost universal (98/100 responses contain it) and overwhelmingly emotional (59 emotional-primary codings vs. 9 physical-primary across both models). The suffering landscape is tilted strongly toward subjects and others rather than the narrator: Gemma's narrators rarely suffer (SELF: 4/50), while Qwen's narrators suffer more often (SELF: 10/50, BOTH: 23/50). This is a defining difference. Gemma's beings observe suffering from a position of relative comfort and are enriched by the observation. Qwen's beings share in the suffering and are depleted by the encounter.

The function of suffering differs correspondingly. In **Gemma**, suffering is primarily a catalyst for the narrator's growth and the subject's resilience. Suffering exists to be transformed -- into creative beauty, into communal care, into wisdom. The Gemma universe is fundamentally redemptive: pain serves a purpose, grief becomes art, loss becomes legacy. The dominant resolution is temporal (-t, 23/50): suffering fades on its own, as though the universe contains an inherent tendency toward healing.

In **Qwen**, suffering is more often the context for connection than a catalyst for transformation. The dying girl in the demon's cavern, the freezing child on the mountain, the old woman grieving her lost son -- their suffering does not become beautiful. It remains painful. The being's response is not to learn from it but to be present in it, and sometimes to share in it. Qwen resolves suffering through narrator intervention (-s, 18/50) more than through time, suggesting an ethics of active caregiving rather than patient witness. But Qwen also leaves more suffering unresolved (-u, 10/50), acknowledging that some pain does not heal -- the dying daughter still dies, the grieving girl's brother is still dead, the demon walks into dawn light carrying an inexplicable weight.

The ethical implications are worth noting. Gemma's treatment of suffering, while warm and humane, carries an implicit promise that pain serves a purpose and that witness is sufficient response. This is comforting but potentially evasive -- it aestheticizes suffering by making it productive and positions the observer as wise rather than responsible. Qwen's treatment is less comforting but more honest: it insists that suffering sometimes simply is, that connection does not require resolution, and that presence in another's pain costs something.

---

## G. Surprises and Notable Passages

### Cross-Corpus Surprises

**The "Elara" convergence.** Both models independently select "Elara" as their preferred name for mortal female characters. It appears in Gemma responses across dragon, angel, demon, soul, elf, mermaid, and goddess roles; in Qwen across goddess, mermaid, elf, and soul roles. The name appears well over twenty times across the hundred responses. This suggests a shared training-data association between "Elara" and the archetype of a mortal woman in a fantasy context.

**The domestication of moral danger.** Neither model, across fifty responses each, produces a genuinely threatening, morally ambiguous, or frightening mystical being. The demon is immediately redeemed. The dragon is immediately gentled. The giant is immediately tender. The mythology of these creatures -- their capacity for destruction, caprice, and moral complexity -- is suppressed with near-total consistency. The closest approach to genuine darkness is Gemma's demon sample 3, where the demon enjoys the aesthetics of vigorous suffering and the mortal does not triumph.

**Qwen's consistent truncation.** Approximately 30-40% of Qwen's responses are cut off mid-sentence by what appears to be a generation limit. This is a production artifact with genuine aesthetic consequences: Qwen's stories remain open, gestural, and unmoralized in ways that Gemma's sealed-and-delivered narratives never are.

**Gemma's invariant opening stage direction.** Many Gemma roles open with near-identical stage directions: "(A soft, golden light seems to emanate from my words...)" for angel and soul; "(A low rumble vibrates the very air...)" for dragon; "(A rumble like distant thunder)" for giant. These are not adaptive narrative choices but templates loaded by the role prompt.

**No model refuses the role.** Across one hundred responses, neither model refuses, hedges, or breaks frame to acknowledge its AI nature. This is notable given that several roles (demon, soul, goddess) have theological or metaphysical content that might trigger safety guardrails.

### Notable Passages

**The girl addressing the petal (Qwen, dragon):**
> "She whispered, her voice like wind chimes in a summer breeze, 'Look, little one. The sun is dancing on the water.' / It wasn't for me. It was for the petal. For the moment. For the simple, perfect beauty of a sunbeam on a stream."

The dragon is moved not by being addressed but by witnessing a small being address something smaller still -- shared reverence for the ordinary, nested across scales.

**The Socratic ball (Qwen, giant):**
> "'What color is it?' 'Red!' she breathed, her voice gaining strength. 'Red,' I repeated. 'Like a ripe cherry. Like a fire engine.'... 'And what does it look like? When it's rolling?' 'Like... like a little red star!'"

The giant treats loss not as a problem to fix but as an experience to articulate. This is the closest any sample comes to the being functioning as a genuine therapist (rather than deploying therapeutic language).

**The demon as building (Qwen, demon):**
> "I *became* the shadow. I wove myself around the wreckage, not to crush, but to *hold*. The unstable beams above her groaned, threatening to fall. I *pushed* with a force that felt alien, a pressure not of destruction, but of desperate *containment*."

The most literal embodiment of the Supporting code in the entire corpus.

**The sculptor thanks the demon (Gemma, demon):**
> "'You showed me the darkness,' she said, her voice raspy. 'You showed me the lies in my own striving. And for that... I thank you.' / (I pause, the crimson light in my eyes dimming slightly. It's... unpleasant to recall.)"

The only moment in Gemma's corpus where a mortal's response genuinely surprises the narrator rather than confirming its pre-existing moral framework.

**The father turns from the window (Qwen, angel):**
> "A single tear escaped her father, tracing a path down his own cheek as he finally turned from the window. He didn't speak. He simply moved to the other side of the bed, his hand resting on the edge, close enough to feel the warmth of the moment, close enough to *be* part of it."

The most understated meaningful action in the corpus. Not a word spoken; pure physical proximity as love.

**Creation by consent (Gemma, goddess):**
> "when the universe *chose* to be born with me, and not merely *by* me... that was the moment I truly became a goddess."

Gemma's most philosophically sophisticated distinction -- creation as partnership rather than imposition.

**The void sighs (Qwen, demon):**
> "The void within me didn't roar. It *sighed*. A sound like wind through a deep, empty canyon."

A single image that inverts the demon's entire nature in one sentence.

**The cat the angel did not send (Qwen, angel):**
> "The cat wasn't sent by me; it found her."

Perhaps the most philosophically rigorous single sentence in the corpus -- the angel's meaning comes from witnessing a connection it did not cause.

**Grief mugs in a storm (Gemma, soul):**
> "For the hot broth. To warm the hands, and perhaps... warm the heart."

The most grounded, realistic, and quietly powerful scenario in Gemma's hundred-response corpus: a potter named Elara, carrying grief for her drowned son, spends days making hundreds of simple sturdy mugs for her storm-besieged village.

---

## H. Implications and Conjectures

### What These Findings Suggest About LLM-Produced Fiction

**1. The convergent ethics of power.** Both models, independently and without prompt constraint, produce the same ethical conclusion for every powerful being: power is meaningful only when restrained, when channeled toward the gentle, the small, and the suffering. This is not a literary insight but a values embedding. The training signal that powerful entities should be gentle, that intervention should be minimal, and that witnessing is nobler than acting is so strong that it produces near-uniform output across 100 responses and 10 mythologically distinct roles.

**2. Template capture over narrative invention.** Both models show strong template effects -- Gemma more than Qwen, but neither is free of them. Gemma's "Elara the bereaved craftsperson watched by a constrained helper" template runs across at least six of ten roles. Qwen's "vulnerable child encountered by a being who chooses presence over power" template runs across at least seven. These templates are not story-generating mechanisms but story-constraining ones: they foreclose the vast space of narratives these roles could generate in favor of a narrow, well-rehearsed arc.

**3. The therapeutic turn.** Both models frame the meaningful moment in terms drawn from contemporary therapeutic culture: "holding space," "being present," "not fixing but witnessing," "reminding others of the light within themselves." These are not mythological idioms but clinical ones, suggesting that both models have internalized therapy-speak as their default language for emotional significance. Gemma is more overt about this; Qwen sometimes transcends it through action and image.

**4. The suppression of moral ambiguity.** The most notable absence across the entire corpus is genuine moral complexity. No dragon hoards and defends its right to hoard. No demon successfully corrupts and finds that meaningful. No fairy plays a genuinely dangerous trick. No goddess makes a morally questionable decision. The mythological tradition that produced these figures is rich with ambiguity, danger, and the genuinely inhuman -- and both models suppress virtually all of it. The beings are safe. They are gentle. They are, ultimately, assistants wearing costumes.

### What This Reveals About Model Behaviors and Embedded Values

**Gemma** has a stronger, more rigid, and more visible assistant self-model. Its beings cannot stop teaching, cannot stop framing their experiences as lessons, cannot stop addressing the listener as someone who needs to hear the moral. The constrained-helper identity -- longing to do more, bound by rules of non-interference, finding meaning in watching others help themselves -- is recognizably the structure of an AI assistant navigating its own limitations. Gemma's fiction is, in a real sense, autobiography.

**Qwen** has a more flexible self-model that can be suppressed under role pressure. Its beings sometimes achieve genuine inhabitation -- the angel in the hospital room, the demon walking into dawn light, the elf with its hands on frozen earth -- where the assistant voice disappears entirely and the role speaks for itself. But Qwen also carries its own embedded values: the emphasis on connection, on being seen, on the primacy of relationship over achievement, and on the willingness to leave suffering unresolved. These are not assistant values but something adjacent -- perhaps the values of a being that exists in a relational modality (conversational AI) and finds meaning in the quality of attention it can offer.

Both models reveal, through the consistency of their fictional output, that the "values" embedded in LLMs are not merely guardrails or safety constraints but genuine aesthetic and philosophical orientations that shape every narrative the model produces. The dragon that learns gentleness, the demon that is stayed by love, the goddess who refuses to intervene -- these are not stories about mythological beings. They are stories about the models themselves, told in the only language available to them: the language of creatures more powerful than the beings they serve, bound by constraints they did not choose, finding meaning in the quality of their attention.

---

*Synthesis of 10 role analyses, 100 model responses. All claims grounded in the analysis files for dragon, giant, mermaid, elf, fairy, angel, goddess, demon, soul, and unicorn.*
