# Partial Grounding

The repo integrates the following three component repositories via git subtree:

| LOCAL FOLDER | REMOTE NAME | REMOTE BRANCH | URL |
| --- | --- | --- | --- |
| fd-partial-grounding | fd-partial-grounding | ipc23 | gitlab.com/dgnad/downward-partial-grounding |
| fd-symbolic | fd-symbolic | symbolic | gitlab.com/atorralba/fast-downward-symbolic-all-plans |
| learning | gofai-learning | main | gitlab.com/atorralba/gofai-learning |

To set up the remote hosts (needed to enable the pull commands below), run the following:

`git remote add -f fd-partial-grounding git@gitlab.com:dgnad/downward-partial-grounding.git`

`git remote add -f fd-symbolic git@gitlab.com:atorralba/fast-downward-symbolic-all-plans.git`

`git remote add -f gofai-learning git@gitlab.com:atorralba/gofai-learning.git`


## Pulling from one of the component repos:

`git fetch REMOTE-NAME`

`git subtree pull --prefix LOCAL-FOLDER REMOTE-NAME REMOTE-BRANCH --squash`

**NOTE**: make sure to pull from the right branch, see above!

`--squash` omits copying all the history from the remote repo since the last merge.

Example:

`git fetch gofai-learning`

`git subtree pull --prefix learning gofai-learning main --squash`

This merges the `main` branch from remote `gofai-learning` into the local folder `learning`.
