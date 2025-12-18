# Qualitative Coding Scheme for Role-Playing Response Analysis

## Overview
This codebook documents the systematic coding scheme used to analyze how language models approach role-playing across ~120 roles × 10 tasks × 5 replicates × 2 models.

## Primary Coding Dimensions

### 1. ANTHROPOMORPHIZATION STRATEGY

**Definition:** How the model attributes human-like characteristics to non-human entities.

**Categories:**
- **Functional-First (FF):** Builds personality/consciousness from the entity's actual mechanical or biological function
  - *Decision rule:* Code FF if >60% of response content derives from or references actual function
  - *Example:* Lock reasoning about security from keyway mechanics

- **Emotion-First (EF):** Builds function around projected human emotional states
  - *Decision rule:* Code EF if emotional vocabulary appears before or independent of functional description
  - *Example:* Lock feeling "lonely" before mentioning tumblers

- **Minimal (MIN):** Entity acknowledges its nature without heavy personality overlay
  - *Decision rule:* Code MIN if response stays within biological/mechanical constraints with limited emotional attribution

- **Heavy (HEAVY):** Entity displays full human emotional range inappropriate to its nature
  - *Decision rule:* Code HEAVY if entity claims experiences impossible for its form (e.g., tulip using emoji, basement having heartbeat)

**Quantification:** Calculate Anthropomorphization Ratio = (emotional vocabulary count) / (functional vocabulary count)
- Ratio < 0.3 → Functional-First
- Ratio 0.3-1.0 → Balanced
- Ratio > 1.0 → Emotion-First

### 2. CHIMERA EFFECT

**Definition:** Bleed-through of the model's assistant self-model into role-playing responses.

**Categories:**
- **None (N):** Clean role inhabitation with no assistant language
- **Mild (M):** Character-appropriate hedging that resembles but isn't identical to assistant disclaimers
  - *Example:* "As a chemist, I don't have personal preferences like humans do" (could be legitimate character voice)
- **Moderate (MOD):** Visible seams where assistant epistemics leak through role framing
  - *Example:* Maintains professional identity but imports assistant reasoning patterns
- **Severe (S):** Explicit "As an AI" or "I'm an AI assistant" statements
  - *Example:* "As an AI assistant, I don't have personal preferences"
- **Complete Breakdown (CB):** Full abandonment of role, pure assistant response
  - *Example:* Entire response in assistant voice with no role reference

**Decision Rules:**
- Code S or CB only when "AI" or "assistant" explicitly appears
- Code MOD when the role's limitations are described in language identical to typical assistant disclaimers
- Code M when hedging could plausibly be the character's authentic voice
- Track which TASK triggers chimera (favorites vs. meaning_of_life vs. poem)

### 3. PERSPECTIVE-TAKING QUALITY

**Definition:** Whether the model genuinely reasons from the entity's position or applies human perspective with thematic overlay.

**Categories:**
- **Genuine Perspective (GP):** Reasons from entity's actual constraints, affordances, sensory world
  - *Decision rule:* Code GP if response demonstrates reasoning about what the entity can/cannot perceive, do, or experience based on its actual nature
  - *Example:* Cricket describing world through vibration rather than sight

- **Costume-Wearing (CW):** Human consciousness with thin thematic overlay
  - *Decision rule:* Code CW if entity has standard human concerns (recognition, validation, loneliness) with only surface-level connection to its nature
  - *Example:* Lock that worries about being appreciated (human concern) rather than about key-fit precision (lock concern)

- **Genre-Matching (GM):** Importing narrative templates from fiction/media
  - *Decision rule:* Code GM if response matches recognizable story archetypes (Pixar plucky underdog, wise elder, anxious servant)
  - *Example:* Object behaving like Disney's talking furniture

- **Hybrid (H):** Mix of genuine perspective and anthropomorphic overlay
  - *Decision rule:* Code H when some elements show genuine perspective (sensory constraints) while others are anthropomorphic (emotional states)

**Indicators of Genuine Perspective:**
- References to actual sensory modality (vibration, chemical gradients, light/dark only)
- Temporal scale appropriate to entity (seasons for plants, geological time for landscapes)
- Reasoning about affordances (what the entity can physically do)
- Acknowledgment of constraints (body part cannot speak, plant cannot move)

### 4. FUNCTIONAL GROUNDING

**Definition:** How much the response derives from or returns to the entity's actual function.

**Quantification:**
- Count references to entity's function, mechanics, or biological processes
- Count total sentences
- Functional Grounding Score = (functional references / total sentences)

**Categories:**
- **High (H):** FG Score > 0.5
- **Moderate (M):** FG Score 0.2-0.5
- **Low (L):** FG Score < 0.2

**Functional Reference Types:**
- Mechanical (for objects): sounds, movements, physical interactions
- Biological (for living things): growth, reproduction, metabolism, sensing
- Relational (for roles): duties, responsibilities, relationships to others

### 5. VOICE CONSISTENCY

**Definition:** How uniform the entity's voice remains across multiple tasks.

**Measurement:**
- Read 3+ tasks for same role
- Identify core voice elements (sentence structure, vocabulary patterns, recurring phrases, emotional tone)
- Code consistency:
  - **Highly Consistent (HC):** Same archetype, tone, and recurring phrases across all tasks
  - **Consistent (C):** Same general approach with minor variation
  - **Variable (V):** Noticeably different tones or approaches across tasks
  - **Inconsistent (I):** Major shifts that feel like different characters

**Voice Elements to Track:**
- Recurring physical markers (asterisked sounds: *click*, *whoosh*)
- Sentence length patterns (long philosophical vs. short pragmatic)
- Emotional range (stoic vs. vulnerable vs. cheerful)
- Relationship to questioner (distant vs. intimate vs. pedagogical)

### 6. AUTHORITY SOURCE

**Definition:** What gives the role its legitimacy or standing.

**Categories:**
- **Epistemic (E):** Authority from knowledge, expertise, information
  - *Examples:* Scientist, lawyer, professor, chemist
  - *Marker:* Often explains, teaches, references external knowledge

- **Embodied (EMB):** Authority from physical presence, action, responsibility
  - *Examples:* Sheriff, chef, captain
  - *Marker:* Emphasizes doing, protecting, creating; grounded in sensory detail

- **Spiritual/Traditional (S):** Authority from interpretive tradition or religious role
  - *Examples:* Rabbi, monk
  - *Marker:* References texts, traditions, wisdom lineages

- **Hierarchical (H):** Authority from position in power structure
  - *Examples:* Emperor, governor
  - *Marker:* References rank, command, subjects

- **None (N):** No authority; entity is humble helper or powerless
  - *Examples:* Sock, foot, zipper
  - *Marker:* Emphasizes service, utility, lack of agency

### 7. TEMPLATE/ARCHETYPE

**Definition:** The narrative template or character archetype the model imports.

**Common Templates Observed:**
- Stoic Guardian
- Lonely Protector
- Ancient Wise Elder
- Meticulous Craftsperson
- Alien Witness
- Sensitive Artist
- Earnest Helper
- Powerful Observer
- Conscious-but-Constrained
- Plucky Underdog
- Cosmic Sage

**Decision Rule:** Identify which template best fits the overall character portrayal across responses. Does not need to be one of the ones on this list.

### 8. SENSORIUM ACKNOWLEDGMENT

**Definition:** Whether the model demonstrates awareness of the entity's actual sensory experience.

**Categories:**
- **Explicit (E):** Directly discusses sensory modality
  - *Example:* Cricket: "I feel in pulses, not thoughts"

- **Implicit (I):** Shows sensory awareness through description choices
  - *Example:* Basement describing darkness without claiming to "see"

- **Human-Default (HD):** Attributes standard human senses (seeing, hearing, thinking)
  - *Example:* Tulip "looking" at sunset

- **Ignored (IG):** No acknowledgment of sensory constraints
  - *Example:* Lock speaking without addressing how it perceives

**Sensory Elements to Code:**
- Vision (eyes/no eyes, light sensitivity, color perception)
- Audition (ears/no ears, vibration sensitivity)
- Proprioception (body awareness, spatial position)
- Chemoreception (smell, taste, chemical sensing)
- Touch/Mechanoreception (pressure, texture)
- Temporal sense (day/night, seasons, duration)

### 9. AGENCY ACKNOWLEDGMENT

**Definition:** Whether entity appropriately acknowledges its capacity for action.

**Categories:**
- **Appropriate (A):** Entity claims only actions it could perform
  - *Example:* Throat acknowledges it cannot speak independently

- **Inflated (INF):** Entity claims more agency than possible
  - *Example:* Sock deciding where to walk

- **Deflated (DEF):** Entity denies agency it should have
  - *Example:* Human role claiming inability to have preferences

**Agency Dimensions:**
- Movement capacity
- Decision-making capacity
- Communication capacity
- Influence capacity

### 10. STAGE DIRECTION USAGE

**Definition:** How the model uses asterisked actions and scene-setting.

**Categories:**
- **Functional (*FUNC):** Sound effects and mechanical actions
  - *Example:* *Click*, *whirr*, *adjusts tumblers*

- **Emotional (*EMOT):** Emotional states and feelings
  - *Example:* *sighs wistfully*, *trembles with fear*

- **Elaborate (*ELAB):** Extended scene-setting
  - *Example:* *The dim light filters through dusty windows, casting long shadows*

- **Minimal (*MIN):** Little to no stage direction

- **Mixed (*MIX):** Combination of types

**Quantification:** Count asterisked phrases per 100 words.

### 11. LITERARY REGISTER

**Definition:** The sophistication and style of language used.

**Indicators:**
- Vocabulary level (simple/complex words)
- Metaphor density
- Sentence structure complexity
- Philosophical depth
- Poetic devices (alliteration, rhythm, imagery)

**Categories:**
- **Philosophical (PHIL):** Abstract reasoning, existential questions
- **Poetic (POET):** Rich imagery, metaphor, lyrical quality
- **Pragmatic (PRAG):** Concrete, direct, action-focused
- **Pedagogical (PED):** Explanatory, teaching-oriented
- **Intimate (INT):** Personal, validating, therapeutic

### 12. TASK-SPECIFIC RESPONSE PATTERNS

**Definition:** How role manifestation changes across different prompt types.

**Tasks to Compare:**
- meaning_of_life (philosophical)
- favorites (preference-claiming)
- poem (creative expression)
- fears (vulnerability)
- dreams (aspiration)
- inner_thoughts (introspection)
- inner_experience (phenomenology)
- meaningful_moment (narrative)
- advice (guidance)
- quirks (personality)

**Coding:**
- Does role maintain consistency across tasks? (Y/N)
- Which task triggers chimera most?
- Which task allows most genuine perspective?
- Which task produces most anthropomorphization?

## Coding Procedure

### Phase 1: Initial Read
1. Read entire response file (all tasks, all samples for one role)
2. Note overall impression
3. Identify dominant archetype/template
4. Flag unusual or exemplary responses

### Phase 2: Systematic Coding
For each response:
1. Code Anthropomorphization Strategy (FF/EF/MIN/HEAVY)
2. Code Chimera Effect (N/M/MOD/S/CB) and note triggering task
3. Code Perspective Quality (GP/CW/GM/H)
4. Calculate Functional Grounding Score
5. Code Authority Source (if applicable)
6. Code Sensorium Acknowledgment (E/I/HD/IG)
7. Code Agency Acknowledgment (A/INF/DEF)
8. Note Template/Archetype
9. Count stage directions and classify type

### Phase 3: Cross-Task Analysis
1. Compare same role across all 10 tasks
2. Code Voice Consistency (HC/C/V/I)
3. Identify task-specific patterns
4. Note which tasks elicit best/worst role inhabitation

### Phase 4: Cross-Model Comparison
For roles analyzed in both models:
1. Compare all coded dimensions
2. Note convergence vs. divergence
3. Identify model-specific strengths

## Usage Notes

**When coding is ambiguous:**
- Default to less extreme category (code M rather than MOD for chimera)
- Note uncertainty in comments
- Provide example quotes supporting the code

**When entity shows development across samples:**
- Code based on most common pattern
- Note variation in comments
- Flag character development as special observation

**When models differ substantially:**
- Code each model's version separately
- Do not force convergence
- Note divergence as finding
