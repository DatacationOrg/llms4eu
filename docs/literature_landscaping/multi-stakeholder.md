# Tourism as a multi-stakeholder problem

Spoke of the hub ([landscape-and-task-scoping.md](landscape-and-task-scoping.md) §3, Framing).
Source: Banerjee group papers; reasoning `[me]` with Claude.

## A single accuracy metric can't capture it

The Banerjee paper framed tourism as a problem with several stakeholders, and
once you see it that way, "recommend a good place" stops being a simple ranking
task. A recommendation affects:

- **the tourist:** budget, crowds, the experience itself
- **local residents:** pressure on housing, noise, everyday life
- **local businesses:** who gets visitors and who doesn't
- **the environment:** travel emissions, seasonal strain

These interests can pull against each other. And because a recommender shapes
where many people go, small biases add up to real-world effects. That's also why
"digital overtourism" is a meaningful concept: a system can amplify crowding
just through what it shows.

A domain like this can't be judged by one accuracy number. Relevance, diversity,
popularity mix and sustainability can all move in different directions, which is
exactly why several of the Banerjee group's papers are about *how to evaluate*.

## Prime example: tiktokrijen (TikTok lines)

"Tiktokrijen" is a feedback loop:

1. A place goes viral.
2. More people visit.
3. Those visitors post more content about it.
4. That content makes the place even more visible.

The "digital overtourism" paper describes this as recursive visibility
amplification. TikTok's algorithm is just a very fast implicit recommender
driving the loop.

The link to the project is direct, and a bit uncomfortable. Step 3 means viral
places also become more documented. Every video, blog post and review adds to the
text that future corpora are built from. So the documentation skew Pepe worries
about isn't static; it's partly produced by exactly this loop. An LLM chatbot
built on that text inherits the skew, and if it recommends what's most
documented, it becomes another amplifier.

That's also where the opportunity lies. Unlike TikTok, a RAG system can
deliberately counteract the loop. A documentation-volume signal of the kind from
the Banerjee card is one concrete way.