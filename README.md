# Partial Grounding

The repo integrates the following three component repositories via git subtree:

[LOCAL FOLDER] [REMOTE NAME] [URL]

- fd-partial-grounding, fd-partial-grounding, gitlab.com/dgnad/downward-partial-grounding
- fd-symbolic, fd-symbolic, gitlab.com/atorralba/fast-downward-symbolic-all-plans
- learning, gofai-learning, gitlab.com/atorralba/gofai-learning

## Pulling from one of the component repos:

Pick one of the repos: fd-partial-grounding, fd-symbolic, gofai-learning

`git fetch gofai-learning`

`git subtree pull --prefix learning gofai-learning main --squash`

This merges the `main` branch from remote `gofai-learning` into the local folder `learning`.

`--squash` omits copying all the history from `gofai-learning` since the last merge.
