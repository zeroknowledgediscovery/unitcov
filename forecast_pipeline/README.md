# Prediction
In the `unitcov/forecast_pipeline` folder, run notebook in exactly the following order
1. Run `pipeline_data_gathering.ipynb`
2. Run `pipeline_GLM.ipynb`
3. Run `pipeline_forecast_case.ipynb` and `pipeline_forecast_death.ipynb`

# Do varification
`pipeline_forecast_[target_type].ipynb` has varification coded in for the _data_ file. 
But the meta data need varification too.

For local validation, follow the instruction here: 
I suggest that we do varification after adding the submission (and/or modifying meta file) *in* the `covid-forecast-hub` repo, by running `python3 code/validation/test-formatting.py`.
It will run validation for both data and meta file. 
(But don't use `add .` in the `push` step below. Just add your file, since validation also produces some log file that may cause conflict.)

## Resource on creating a pull reqeust:
https://opensource.com/article/19/7/create-pull-request-github

# Submission Steps:
1. Fork the the [reichlab/covid19-forecast-hub](https://github.com/reichlab/covid19-forecast-hub) repo (*only for the first time*); 
   **How**: on the repo webpage, click on the `Fork` button in the top-right corner. 
   This creates a new copy of repo under your GitHub user account with a URL like:
   `https://github.com/<YourUserName>/covid19-forecast-hub`.
2. Clone it to your local system (*only for the first time*) ;
    **How**: Open a terminal, move to a location where you want to keep your local version of the repo, run bash comment `git clone https://github.com/<YourUserName>/covid-forecast-hub`
3. Make a new branch (*only for the first time*);
    **How**: Inside you local repo, run bash comment `git checkout -b [new_branch]`. 
    `[new_branch]` is a name you should pick up (carefully), such as `zed_submission_branch`. 
    Don't just use `new_branch` literally.
4. Make your changes:
    **How**: we do the prediction in the `unitcov` repo. 
    After we get the submission file in the `submission` folder, 
    we just need to add the file to the `data-processed/UChicagoCHATTOPADHYAY-UnIT` folder.
    (We may want to make due modifications to the meta file that is already in the folder).
    **Don't forget to run validation!** (discussed above)
5. Push it back to your repo;
    **How**: 
    - `git add [forecast_datafile] [meta file]`;
    - `git commit -m "UChicagoCHATTOPADHYAY-UnIT forecast on [date]"`
    - `git push`
6. Click the `Compare & pull request` button (only for the first time), then just follow the instruction on the webpage;
    For the second time on:
    - in `reichlab/covid19-forecast-hub`, click the `Pull request` button;
    - click the green `New pull request` button;
    - choose your fork;
    - choose your branch;
    - Then just following the instruction on the webpage.
