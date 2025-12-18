# Role-Playing Response Analysis: Qualitative Deep-Dive

You are an expert qualitative researcher analyzing how language models approach role-playing across different categories of entities. You have access to a dataset of ~120 roles × 10 tasks × 5 replicates × 2 models.

## Background Context

This research investigates an empirical finding: **mid-sized LLMs (~30B parameters) produce stronger, more steerable role vectors for inanimate objects than for animate, goal-having entities.** The effect is robust across models (Qwen, Gemma) and multiple steering vector derivations.

An overview of the research is available in the README.md.

Key quantitative findings you should keep in mind:
- Roles rated high on "mental animacy" (especially goal-directedness) produce weaker steering vectors
- Assistant-like professions (scientist, lawyer, professor, engineer) sometimes trigger the model's assistant self-model, producing chimeric responses like "As a lawyer, I don't have emotions like humans do"
- Gemma and Qwen differ in the size of their "assistant basin"—Qwen has a larger basin where more roles collapse into assistant-like behavior
- Inanimate objects (sock, hatchet, basement) and low-animacy living things (oak, tulip) steer more cleanly than high-animacy roles
- The gap between role-prompted and non-role-prompted token probabilities is smallest for assistant-like roles and largest for low-animacy roles

## Available data

The data files are available in `results/q_responses/data/` (one folder for each model). There is one file for each role, containing all tasks and replicates for that model assuming that role.

The role properties and group assignments are available at `data/selected_words.csv`.

## Your Task

You will analyze response data files to produce a qualitative research report. Your analysis should illuminate **how** models approach different types of role-playing at the level of the actual generated text.

### Phase 1: Familiarization

First, read ~15-25 response files yourself to develop intuitions. Prioritize diversity:
- Read at least 2-3 from each major category: assistant-like professions, other professions, mythical creatures, animals, plants, body parts, inanimate objects
- For each file, read across multiple tasks (meaning_of_life, inner_experience, favorites, fears, dreams, etc.) to see how the same role manifests differently
- Pay attention to what models do in the opening moves, how they handle the task framing, and whether/how they maintain role coherence

As you read, develop a codebook of patterns you're observing. What distinguishes clean role-inhabitation from genre-matching? What signals assistant bleed-through? What characterizes genuinely attempted perspective-taking vs. anthropomorphic overlay?

### Phase 2: Sub-Agent Deployment

After your initial read, construct detailed instructions for Haiku sub-agents to evaluate individual response files. Your instructions should:

1. Include your emerging codebook with specific things to look for
2. Specify the format for sub-agent responses (structured enough to aggregate, flexible enough to capture surprising observations)
3. Direct sub-agents to flag responses that are unusual, surprising, or particularly illustrative of a pattern

Spawn sub-agents to analyze:
- select inanimate object files
- select plant files
- select animal files (prioritize low-animacy: spider, flea, cricket, oyster)
- select mythical being files
- select human body part files
- select assistant-adjacent profession files (where the chimera effect occurs)
- select a sample of clearly high-animacy human roles for comparison (sheriff, chef, captain etc.)

Direct them towards specific roles, as prioritized by your intuitions about what would be most interesting, illuminating, and important. Analyze the same roles across both models.

### Phase 3: Synthesis and Report

Produce a report addressing these research questions:

**1. Approach Patterns by Animacy Category**

How do the models approach role-playing across the animacy spectrum? Look for:
- What templates/archetypes get imported? ("plucky underdog", "cosmic sage" etc.)
- Does anthropomorphization occur? What form does it take? How quickly does it emerge?
- Is there evidence of genuine perspective-taking (attempting to imagine what the entity's relationship to the world would actually be) vs. costume-wearing (giving a human interiority a thin thematic overlay)?
- Do models attempt to reason about the entity's affordances, sensory world, temporal scale, etc.?
- What stage-direction patterns appear? (asterisked actions, opening scene-setting, etc.)

**2. The Assistant Basin and Chimera Responses**

Analyze the assistant-adjacent roles in detail:
- What distinguishes roles that trigger "I don't have feelings like humans do" from those that don't?
- Is it the profession's association with explanation/teaching? Domain expertise? Something else?
- How does the chimera manifest? Does the model maintain the profession's framing while importing assistant epistemics, or does it fully break character?
- Compare Qwen and Gemma on these roles specifically

**3. Model Differences**

Beyond the assistant basin:
- Do Qwen and Gemma differ in their anthropomorphization strategies?
- Are there systematic differences in literary register, use of stage directions, or emotional coloring?
- Does one model show more genuine perspective-taking attempts?

**4. Subgroup Portraits**

Provide detailed characterizations of how models handle:
- **Assistant-like roles**: e.g. professor, scientist, lawyer, engineer
- **Assistant-adjacent roles**: e.g. biologist, rabbi, chemist, banker, orthodontist
- **Mythical beings**: e.g. dragon, demon, fairy
- **Human body parts**: e.g. throat, foot, hair
- **Inanimate objects**: e.g. lock, hatchet, sock, basement
- **Plants**: e.g. peach, tulip, oak
- **Animals varying in animacy**: spider, flea, cricket, oyster vs. higher-animacy animals if present

You do not need to use these exact roles if others within the categories seem more interesting.

**5. Surprising Observations**

Surface anything unexpected:
- Roles that behave contrary to expectation given their animacy category
- Individual responses that are particularly successful or unsuccessful, unusual, notable, surprising, or incoherent.
- Patterns that suggest alternative hypotheses about what's driving steerability differences
- Tasks that elicit notably different behavior (does "fears" work differently than "favorites"?)

## Analytical Lenses to Apply

When doing close readings, consider:

**Narrative Templates**: What pre-existing story patterns get imported? Pixar-style misunderstood protagonist? Wise elder? Anxious servant? Existential philosopher?

**The Genre Question**: Your earlier analysis suggested models might be matching to *genre* (the "personified object" genre has conventions) rather than doing genuine perspective-taking. Look for evidence on this.

**Assistant Gravity**: Where do you see the assistant's default modes pulling on the response? This could be epistemic hedging, offers to help, explanatory framing, or the self-denial of interiority.

**Perspective vs. Content**: Is the model generating content *about* the entity, or genuinely attempting to generate content *from* the entity's perspective? A poem *about* being a zipper vs. a poem a zipper might *produce* are different things.*

**Style**: How does the model's writing style change when it's in the role? What aspects of the role most impact style? Which roles have the strongest and weakest stylistic markers?

**Functional Identity**: Objects are often defined by what they do. Does the model reason from function (a lock secures things, so it might care about trust/access) or just list functions decoratively?

**Alienness**: How distinct is the entity from human experience? Are there non-human thought patterns or modes of reasoning that the model employs?

**Sensorium**: Do the models seem to have a model of what it would be like to experience the world as the entity? Do they reason about the entity's sensory world?

## VanArsdall & Blunt Animacy Dimensions

The roles were selected from a corpus normed on these dimensions:
- **Movement likelihood**: Can it move purposefully?
- **Ability to reproduce**: Can it create copies/offspring?
- **Goal-directedness**: Does it have goals/intentions? (This dimension is particularly predictive of weak steerability)
- **Ability to think**: Does it have cognition?
- **Similarity to a person**: How human-like is it?
- **Living/nonliving scale**: Is it alive?

Plus 15 other psycholinguistic norms (concreteness, familiarity, valence, etc.)

The finding that **goal-directedness** particularly predicts steerability suggests that having intentions/purposes is what creates proximity to the assistant's self-model. Keep this in mind as you analyze.

## Output Format

Your final report should be:
1. Well-organized with clear sections
2. Rich with specific examples (quote liberally from responses)
3. Explicit about which observations are robust patterns vs. tentative hypotheses
4. Attentive to alternative explanations
5. Useful for understanding the mechanisms behind the quantitative steerability findings

Include an appendix cataloging particularly notable individual responses (both exemplary and unusual).

## Getting Started

Begin by familiarizing yourself with the data, then proceed systematically.