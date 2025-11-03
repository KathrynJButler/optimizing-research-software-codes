# Contributing Guide
- Set Up: Clone repository to local environment. Verify correct dependencies, environmental variables, and configuration files. Confirm local and baseline outputs are matching before editing code. 
- Code: Code edits are expected to be scoped and concise. Only one feature should be changed or fixed per pull request. Follow style guidelines and include detailed comments to ensure the code is readable and reusable. 
- Test: Write unit tests for new functionality to be submitted. All tests will be written in our program’s main testing file. Update pre-existing tests if necessary. Run tests locally and ensure all tests pass with no errors.
- Code Review: Submit pull request with descriptive title and information. At least one team member must review. Reviewing team members are expected to be responsive and  provide constructive feedback.
- Documentation: Summarize changes in the sprint document.
- Final Checklist Before Merge: 1 - code runs correctly on local environment, 2 - outputs match the expected baseline, 3 - all tests pass with no errors, 4 - at least one team member has reviewed and approved the pull request, 5 - ensure code has no linting errors, 6 - document changes are summarized in sprint notes. 

## Code of Conduct
All contributors are expected to follow Oregon State University’s student code of conduct as well as the CS Capstone course standards. 
- Be respectful, responsive, and inclusive in all interactions.
- Provide constructive feedback in reviews.
- Report any concerns to the TA or instructor.

## Getting Started
- Anaconda is highly recommended as an environment handler to make setting up required libraries much easier.
- Create your directory and add the four top-level folders from the Github repository.
- After installing all dependencies, either with or without Anaconda, the code should almost be ready to run.
- Edit the yaml.config file to read the Navigation and Observation files you wish to use.
- In the terminal of the directory /gnss_python-main/ enter ‘python rnx2db.py’ to run the program.
## Branching & Workflow
- Workflow: We use trunk-based development workflow to manage our edits. This keeps our workflow quickly updated and continuously integrated.
- Default Branch: The main branch holds all production-ready code. All added features and fixes are merged frequently. 
- Branching: Create temporary branch off main for each task. Delete after merging to uphold good cleanup.
- Branch Naming:

| Branch Type | Prefix | Example |
|--------------|---------|----------|
| Feature | feature/ | feature/process-file |
| BugFix | bugfix/ | bugfix/unittest-4-error |
| Task | task/ | task/update-conditions |

- Rebase Vs. Merge: Rebase development branch onto most recent main branch before submitting pull request. This will avoid merge conflicts. Merge branch only after passing all quality gates and obtaining peer approval.

## Issues & Planning
Our team uses GitHub Issues to document issues we have and to track bug removal and documentation updates. To file an issue,
1. Go to the Issues tab in the repository.
2. Click “New Issue”.
3. Include:
- A short title of the issue or task
- A description of what’s wrong or what’s being added
- Expected and actual behavior
- Steps to reproduce the issue if applicable
- Links to any code or documentation

Estimations: 
- Small: < 1 hour of work (typo, comment update)
- Medium: 1-3 hours of work (small bug or change)
- Large: 3+ hours of work (major bug or change)
Triage and Assignment Practices: 
- Issues are assessed each Friday during team meetings and/or before each sprint.
- If an issue depends on project partner feedback or authorization, it will be unassigned and left until project partner review.

## Commit Messages
Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format (e.g., <type>(scope): <short summary>). For example, fix(parser): optimized RINEX3 file loading.
Reference issues using #<issue-number>. For example, #123.

## Code Style, Linting & Formatting
- We will be using Pylint as out linter
- Use ’pip install pylint’ to install pylint
- To check for linting issues in our code, we will run whichever function we wish to check with pylint. 
- For example, if we wanted to check the rnx2db.py file we would run ‘pylint rnx2db.py’ in the gnss_python-main directory.
## Testing
- Since this project is inherited and was fully functional at the time we received it, the only test we’ve had to implement so far has been a runtime benchmarking test so we could identify bottleneck functions in the code.
- We implemented a test function decorator called ‘timeit’ which prints accumulated time spent in any function you decorate with “@timeit” at the end of the program.
- New tests may become necessary as we begin making changes to the bottleneck functions we found.
- All functionality that was present when we received the codebase is expected to remain when we’re finished, so testing will be required to maintain that level of coverage.

## Pull Requests & Reviews
Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.
- Pull request requirements: Use standard pull request template .github/PULL_REQUEST_TEMPLATE.md to detail the changes that were made. See “Contributing Guide - Final Checklist Before Merge” as this is identical to our pull request checklist. Attempt to keep pull requests under 50 lines of code changed. Ideally, all changes are done within one function.
- Reviewer Expectations: Provide feedback within 24 hours of when the request is submitted, check correctness and naming conventions, use comments to provide in-line notes.
- Approval rules: One reviewer must approve request before merging, all testing and CI/CD checks must pass, there must be no merge conflicts.
- Required Status Checks: Check unit tests, Check linting and formatting, Check documentation.

## CI/CD
Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs,
and what must pass before merge/release.
## Security & Secrets
We don’t expect that this project will handle sensitive data, but everyone contributing should follow good security practices to protect our code.

To report vulnerabilities, do not open a public issue. Instead, please email our team and Dr. Park (jihye.park@oregonstate.edu) directly with details on the vulnerability. We will address and resolve this security issue and push a public update.
- Never commit hard-coded secrets, keys, or other sensitive information to our repository. 
- In regards to dependencies, we rely on the Python standard library unless otherwise approved. 
- We will run vulnerability checks using the pip audit scanning tool before merging. 

## Documentation Expectations
- Our Github repository contains a detailed README file which we update as changes are committed to the codebase or we have any external changes in direction we wish to reflect in documentation.
- Our docstrings and comments will stay neat and concise while retaining readability and comprehensibility for any future maintenance work or updating that needs to be done by future teams.

## Release Process
Since our project focuses so much on updating an existing research tool (instead of deploying a full product from start to finish), releases are tied to major deliverables such as sprint completions, milestones met, and final submissions.

Our versioning scheme follows vMajor.Minor.Patch (e.g., v1.0.0). A major update refers to an algorithm change, a minor update refers to performance updates and small features introduced, and a patch includes small bug fixes and documentation updates.

After merging our work for the sprint into main, we will create a GitHub release with a tag which matches the version (e.g., v1.2.0). In the release notes and CHANGELOG.md, we will include a summary of the key improvements, test data and screenshots, and updated documentation.

Before publishing, we will verify that:
- All tests pass.
- Documentation and README are updated.
- No temporary or debugging files exist.
- Version change is added to both the code and the changelog.

In the event of a broken release, we will rollback by: 
- Reverting the merge commit by using the “Revert” button.
- Tagging a hotfix version (e.g., v1.2.1-hotfix) to record the rollback.

## Support & Contact
For assistance or to propose improvements, please use GitHub Issues or email. Expect a response within 3-5 business days.
Email:
- Kathryn Butler (Documentation Lead): butlekat@oregonstate.edu
- Michael Mcallister (Development Lead): mcallmic@oregonstate.edu
- Joseph Schaab (Testing & Review Lead): schaabj@oregonstate.edu
