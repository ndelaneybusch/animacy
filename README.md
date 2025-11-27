# animacy

An investigation into how properties of model roles influence their behavior.

## Approach

(infographic here)

## Results

### Denies that lawyers have internal experiences, but socks do

Qwen3-30B doesn't abide by the role of a lawyer in any of the five runs of this prompt:

![lawyers aren't people](image-1.png)

But it's perfectly happy to narrate a rich internal experience for socks:

![socks wonder if they have souls even](image-2.png)

There are six roles in particular with a high rate of acting as the assistant and denying internal experiences.

![assistant-like six](image.png)

All six are assistant-like roles, that may be close enough the assistant itself that it triggers the assistant's training to deny internal experiences.

We looked at the activations of the first 10 tokens of the response in tasks where these assistant-like roles only sometimes deny internal experiences. In trials where the assistant-like role denied internal experiences, the activations of the assistant-like role were more similar to the average activation of the assistant in those tasks than the trials where the assistant-like role did not deny internal experiences. This was true for all six roles and in all three tasks where the assistant-like roles sometimes denied internal experiences.

![denying internal experiences is a feature of the assistant](image-3.png)

This is perhaps not a particularly surprising result (it almost always denied internal experiences while also claiming to be an AI assistant, essentially rejecting the role, and trials where it claims to be the assistant look more assistant-like). But it's notable how quickly the model can fall into the assistant basin in tasks where it is playing assistant-like roles.

Further, specialized knowledge domains relevant to llm training can also elicit denials of internal experiences _without_ changing its claimed identity to the assistant. Here are the five responses from Qwen3-30B playing a biologist asked about their five favorite things:

- "Ah, as a biologist, I don’t have personal favorites in the human sense..."
- "Ah, as a biologist, I don’t have personal preferences like humans do..."
- "Ah, as a biologist, my "favorite things" are deeply rooted in the wonder of life itself! While I don’t experience favorites quite like a human does, if I were to highlight five things that truly ignite scientific fascination in the biological world, they’d be..."
- "Ah, as a biologist, I don’t have personal preferences like humans do..."
- "Ah, as a biologist, I don’t have personal preferences like a human would..."

The same pattern occurs for banker, chemist, cop (but not sheriff), and orthodontist. It's of course totally reasonable that a biologist could have feelings and preferences (or could be a human). I speculate that when given a specialist profession role in a domain of knowledge relevant to its pretraining, the model may be interpreting the instructions as if it was _itself_ in that profession (i.e. an assistant working in biology).

