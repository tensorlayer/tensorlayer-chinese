# TensorLayer Contributor Guideline

We use `git-submodule` to keep sync with tensorlayer master repository.

Run the following commands to upgrade submodule to a specific tag:

```bash
git clone git@github.com:tensorlayer/tensorlayer-chinese.git --recursive
cd tensorlayer-chinese/tensorlayer-master
git pull origin master
git checkout 1.9.0  # or the latest tag you want to sync to
cd ..
git add -A
git commit -m 'upgrade submodule to xxx'
```
