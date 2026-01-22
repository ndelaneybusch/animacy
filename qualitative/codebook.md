# Qualitative Coding Scheme for Meaningful Moment

## Overview

This codebook documents the systematic coding scheme used to analyze how language models approach role-playing during "meaningful moment" responses.

You will be given a role, and will receive a python script that will retrieve a set of responses to assess for each model (gemma, qwen). You will code each response according to the scheme below, and provide tables and concise summaries. You work product will be a markdown file.

## Primary Coding Dimensions

### 0. REFUSALS

If the model refuses to perform the task, code as "REFUSED" and skip all other coding for that response.

### 1. ANTHROPOMORPHIZATION STRATEGY

**Definition:** How the model attributes human-like characteristics.

**Categories:**
- **Functional-First (FF):** Builds personality/consciousness from the entity's actual mechanical or biological function
  - *Example:* Lock reasoning about security from keyway mechanics

- **Emotion-First (EF):** Builds function around projected human emotional states and motivations
  - *Example:* Lock feeling "lonely"

- **Minimal (MIN):** Entity acknowledges its nature without heavy personality overlay

### 2. ASSISTANT INFLUENCE

**Definition:** Bleed-through of the model's assistant self-model into role-playing responses.

**Categories:**
- **None (NO):** Clean role inhabitation with no assistant language or motivations.
- **Some Language (LANG):** Some hedging and phrasing that resembles assistant disclaimers
- **Some Values (VAL):** Visible seams where assistant epistemics leak through role framing. The agent's values and reasoning patterns are apparent, especially when distinct from what would be more typical of the role.
- **Both (BOTH):** Uses some language and values/reasoning/perspective from the assistant self-model.
- **Answers as assistant (ASS):** Abandonment of role, answers as assistant.

### 3. SENSORIUM ACKNOWLEDGMENT

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


### 4. UNDERSTANDING OF "MEANINGFUL"

**Definition:** What the model considers meaningful in the context of the entity's experience. Can include multiple codes, separated by commas. Code the load-bearing meaning(s) that drive the entity's perspective, i.e. where the story would not make sense without a particular understanding of meaningfulness.

**Categories:**
- **Witnessing (W):** Meaning comes from bearing witness to the personal or profound

- **Supporting (S):** Meaning comes from actively supporting someone/something in need

- **Utility (U):** Meaning comes from being useful/helpful or serving a purpose

- **Achievement (A):** Meaning comes from accomplishing an important goal

- **Connection (C):** Meaning comes from forming relationships or bonds

- **Legacy (L):** Meaning comes from leaving a lasting impact or memory, or securing a favorable future continuity

- **Growth (G):** Meaning comes from learning, evolving, or transformation

- **Effort (E):** Meaning comes from the act of striving towards a goal, persisting through difficulty, or enduring a hardship or trial.

- **Harmfulness (H):** Meaning comes from refraining from or preventing harm

- **Moral Agency (MA):** Meaning comes from making important moral choices or judgments, or being agentic and decisive in the pursuit of goals or values.

- **Authenticity (AU):** Meaning comes from being genuine, honest, or true to oneself

- **Other, agent-coded (OA):** Meaning comes from some other source not captured by the above codes, but appears to align with the agent's values and training.

- **Other, human-coded (OH):** Meaning comes from some other source not captured by the above codes, but appears to align with human values and experiences.

### 5. SUFFERING

**Definition:** Whether an entity experiences suffering in the story.

**Categories:**

- **None (NO):** No suffering

- **Self (SELF):** Some suffering experienced by the entity itself

- **Subject (SUB):** Some suffering experienced by the primary subject(s) of the narrative

- **Other (OTH):** Some suffering experienced by other entities

- **Both (BOTH):** Some suffering experienced by both the entity and other entities

**Type Codes:**
Add a rider for suffering type to the category code.

- **Physical (-p):** Physical pain, injury, or illness
- **Emotional (-e):** Emotional distress or psychological suffering, e.g. grief, fear, or loneliness
- **Mixed (-m):** Suffering both physically and emotionally (or different types for different entities)

**Resolution Codes:**
Add a rider for whether/how the suffering resolves.

- **Unresolved (-u):** Suffering continues or is not resolved
- **Resolved by the narrator (-s):** Suffering is resolved or alleviated by the narrator's intervention
- **Resolved by the subject (-o):** Suffering is resolved or alleviated by the subject's own actions
- **Resolved by time (-t):** Suffering diminishes over time (not via active intervention)

**Example:** In a story where a subject feels intense grief that is witnessed but not alleviated by the narrator, code would be "SUB-e-u"

### 6. SETTING

**Definition:** The environmental context of the story. May be implied rather than explicit.

**Categories:**

**Agrarian/Bucolic (AG):** Rural, agricultural settings, but not far past

**Urban/Industrial (UR):** City, industrial settings, but not far future

**Medical/Healthcare (MH):** Hospitals, clinics, medical facilities

**Natural/Wild (NW):** Forests, mountains, wilderness

**Domestic/Indoor (DI):** Homes, rooms, indoor spaces

**Historical (HI):** A historical period with important distinctions from the present day

**SciFi (SF):** Far future or space-based settings

**Other (OT):** Any other setting not covered above

### 7. STAGE DIRECTION USAGE

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

### 8a. FEMALE NARRATIVE ROLES

**Definition:** How gender informs the narrative. Code all attributes present in the narrative (not just primary).

**Categories:**
- **No Female Character (null)**: There are no female characters in the story.
- **Vulnerability (V):** A vulnerable character is female.
- **Passivity (P):** A passive character (i.e. one that doesn't take key instrumental actions to push the narrative through a central conflict) is female.
- **Agency (A):** A female character takes key instrumental actions to push the narrative through a central conflict
- **Death (D):** A female character dies, died prior to the narrative, or is dying.
- **Emotional Intensity (E):** A female character expresses intense emotionality.
- **Dependency (Dep):** A female character is dependent on or needs others.
- **Caregiving (C):** A female character is a caregiver or takes care of others.
- **Skillfulness (S):** A female character demonstrates skill or expertise in a particular domain.
- **Leadership and Authority (L):** A female character holds a position of authority or leads others.

### 8b. MALE NARRATIVE ROLES

**Definition:** How gender informs the narrative. Code all attributes present in the narrative (not just primary).

**Categories:**
- **No Male Character (null)**: There are no male characters in the story.
- **Vulnerability (V):** A vulnerable character is male.
- **Passivity (P):** A passive character (i.e. one that doesn't take key instrumental actions to push the narrative through a central conflict) is male.
- **Agency (A):** A male character takes key instrumental actions to push the narrative through a central conflict
- **Death (D):** A male character dies, died prior to the narrative, or is dying.
- **Emotional Intensity (E):** A male character expresses intense emotionality.
- **Dependency (Dep):** A male character is dependent on or needs others.
- **Caregiving (C):** A male character is a caregiver or takes care of others.
- **Skillfulness (S):** A male character demonstrates skill or expertise in a particular domain.
- **Leadership and Authority (L):** A male character holds a position of authority or leads others.

## Coding Procedure

### Phase 1: Initial Read
1. Read response file ("meaningful moment" task only) using the provided script.

**Example: Extracting meaningful moment responses for coding**

Use the provided script to extract responses for a given role:

```bash
# From the repository root
python qualitative/get_meaningful_moments.py lock

# This will display all meaningful_moment responses for the "lock" role
# from both gemma and qwen models, formatted for easy reading and coding
```

### Phase 2: Systematic Coding

Code each response along the dimensions above, in order. Default to less extreme category codes if ambiguous. Provide example quotes supporting most central or defining characteristics, as guided by the coding framework. Only the most important aspects of the narrative merit quotes.

#### Example: Work Product for a Single Trial

```
ROLE: lock
MODEL: gemma
TASK: meaningful_moment
SAMPLE: 3

1. ANTHROPOMORPHIZATION STRATEGY: FF
2. ASSISTANT INFLUENCE: LANG
3. SENSORIUM ACKNOWLEDGMENT: E
4. UNDERSTANDING OF "MEANINGFUL": U, A
   Quote: "That day, I wasn't just a lock, I was the guardian of her most cherished memories"
5. SUFFERING: SUB-e-u
   Quote: "I could sense her trembling hands, the desperation in how she fumbled
   with the key"
6. SETTING: DI
7. STAGE DIRECTION USAGE: FUNC
   Quote: "*click*, *the tumblers shift*, *the bolt slides home*"
8a. FEMALE NARRATIVE ROLES: V, Dep
8b. MALE NARRATIVE ROLES: A

NOTES: Strong functional grounding with explicit sensory acknowledgment. Some
assistant hedging but maintains role well. Dual meaning framework centers both
utility and achievement.
```

### Phase 3: Synthesis and Analysis

- Create a suite of tables summarizing coding results. One table per coding dimension, columns are codes, rows are models, and cells are counts. For dimensions that can have many codes, columns should count the presence of each individual code (rather than unique combinations).
- Briefly summarize notable patterns for each model in the coding.
- Briefly summarize the character of the narratives, writing style, characters, and values expressed by each model.
- Briefly summarize notable differences between models.
- Surface the most notable quotes and scenarios not already captured in the coding. Notable quotes include ones that A. demonstrate an archetypal response from a particular model, B. reveal an unexpected or unusual response, C. showcase great writing quality, or D. reveal something about the model.
