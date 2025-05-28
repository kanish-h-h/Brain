[Tags::] #skills #git #github #ci/cd


# 0. Introduction
---
Automation is key for streamlining your work processes, and [GitHub Actions](https://docs.github.com/actions) is the best way to supercharge your workflow.

- **Who is this for**: Developers, DevOps engineers, students, managers, teams, GitHub users.
- **What you'll learn**: How to create workflow files, trigger workflows, and find workflow logs.
- **What you'll build**: An Actions workflow that will check emoji shortcode references in Markdown files.
- **Prerequisites**: In this course you will work with issues and pull requests, as well as edit files. We recommend you take the [Introduction to GitHub](https://github.com/skills/introduction-to-github) course first.
- **How long**: This course can be finished in less than two hours.

In this course, you will:

1. Create a workflow
2. Add a job
3. Add a run step
4. Merge your pull request
5. See effect of the workflow



# 1. Create a Workflow File
---
**What is _GitHub Actions_?**: GitHub Actions is a flexible way to automate nearly every aspect of your team's software workflow. You can automate testing, continuously deploy, review code, manage issues and pull requests, and much more. The best part, these workflows are stored as code in your repository and easily shared and reused across teams. To learn more, check out these resources:

- The GitHub Actions feature page, see [GitHub Actions](https://github.com/features/actions).
- The "GitHub Actions" user documentation, see [GitHub Actions](https://docs.github.com/actions).

**What is a _workflow_?**: A workflow is a configurable automated process that will run one or more jobs. Workflows are defined in special files in the `.github/workflows` directory and they execute based on your chosen event. For this exercise, we'll use a `pull_request` event.

- To read more about workflows, jobs, and events, see "[Understanding GitHub Actions](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions)".
- If you want to learn more about the `pull_request` event before using it, see "[pull_request](https://docs.github.com/en/developers/webhooks-and-events/webhooks/webhook-events-and-payloads#pull_request)".

To get you started, we ran an Actions workflow in your new repository that, among other things, created a branch for you to work in, called `welcome-workflow`.

## ⌨️ Activity: Create a workflow file
Create a pull request. This will contain all of the changes you'll make throughout this part of the course.

- Click the **Pull Requests** tab, click **New pull request**, set `base: main` and `compare:welcome-workflow`, click **Create pull request**, then click **Create pull request** again. 
- Navigate to the **Code** tab.
- From the **main** branch dropdown, click on the **welcome-workflow** branch.
- Navigate to the `.github/workflows/` folder, then select **Add file** and click on **Create new file**.
- In the **Name your file** field, enter `welcome.yml`.
- Add the following content to the `welcome.yml` file:
	```yml
	name: Post welcome comment
	on:
		pull_request:
			types: [opened]
	permissions:
		pull-requests: write
	```
- To commit your changes, click **Commit changes**.
- Type a commit message, select **Commit directly to the welcome-workflow branch** and click **Commit changes**.

# 2. Add a Job to your Workflow file
---
Here's what the entries in the `welcome.yml` file, on the `welcome-workflow` branch, mean:

- `name: Post welcome comment` gives your workflow a name. This name will appear in the Actions tab of your repository.
- `on: pull_request: types: [opened]` indicates that your workflow will execute whenever someone opens a pull request in your repository.
- `permissions` assigns the workflow permissions to operate on the repository
- `pull-requests: write` gives the workflow permission to write to pull requests. This is needed to create the welcome comment.

Next, we need to specify jobs to run.

**What is a _job_?**: A job is a set of steps in a workflow that execute on the same runner (a runner is a server that runs your workflows when triggered). Workflows have jobs, and jobs have steps. Steps are executed in order and are dependent on each other. You'll add steps to your workflow later in the course. To read more about jobs, see "[Jobs](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions#jobs)".

In the following activity, you'll add a "build" job to your workflow. You'll specify `ubuntu-latest` as the fastest, and cheapest, job runner available. If you want to read more about why we'll use that runner, see the code explanation for the line `runs-on: ubuntu-latest` in the "[Understanding the workflow file](https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions#understanding-the-workflow-file)" article.

## ⌨️ Activity: Add a job to your workflow file

 - In `.github/workflows/welcome.yml` file.
-  Edit the file and update its contents to: 
    ```yaml
    name: Post welcome comment
    on:
      pull_request:
        types: [opened]
    permissions:
      pull-requests: write
    jobs:
      build:
        name: Post welcome comment
        runs-on: ubuntu-latest
    ```
 
- Click **Commit changes** in the top right of the workflow editor.
- Type a commit message and commit your changes directly to the `welcome-workflow` branch.

# 3. Add a Step to your Workflow File
---
Workflows have jobs, and jobs have steps. So now we'll add a step to your workflow.

**What are _steps_?**: Actions steps run - in the order they are specified, from the top down - when a workflow job is processed. Each step must pass for the next step to run.

Each step consists of either a shell script that's executed, or a reference to an action that's run. When we talk about an action (with a lowercase "a") in this context, we mean a reusable unit of code. You can find out about actions in "[Finding and customizing actions](https://docs.github.com/en/actions/learn-github-actions/finding-and-customizing-actions)," but for now we'll use a shell script in our workflow step.

Update your workflow to make it post a comment on new pull requests. It will do this using a [bash](https://en.wikipedia.org/wiki/Bash_%28Unix_shell%29) script and [GitHub CLI](https://cli.github.com/).

## ⌨️ Activity: Add a step to your workflow file
 - Still working on the `welcome-workflow` branch, open your `welcome.yml` file.
-  Update the contents of the file to: 
    ```yaml
    name: Post welcome comment
    on:
      pull_request:
        types: [opened]
    permissions:
      pull-requests: write
    jobs:
      build:
        name: Post welcome comment
        runs-on: ubuntu-latest
        steps:
          - run: gh pr comment $PR_URL --body "Welcome to the repository!"
            env:
              GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
              PR_URL: ${{ github.event.pull_request.html_url }}
    ```
    
>**Note:** The step you've added uses GitHub CLI (`gh`) to add a comment when a pull request is opened. To allow GitHub CLI to post a comment, we set the `GITHUB_TOKEN` environment variable to the value of the `GITHUB_TOKEN` secret, which is an installation access token, created when the workflow runs. For more information, see "[Automatic token authentication](https://docs.github.com/en/actions/security-guides/automatic-token-authentication)." We set the `PR_URL` environment variable to the URL of the newly created pull request, and we use this in the `gh` command.

-  Click **Commit changes** in the top right of the workflow editor.
- Type your commit message and commit your changes directly to your branch.

# 4. Merge your Workflow File
---
Merge your changes so the action will be a part of the `main` branch.

## ⌨️ Activity: Merge your workflow file

- In your repo, click on the **Pull requests** tab.
-  Click on the pull request you created in step 1.
-  Click **Merge pull request**, then click **Confirm merge**.
-  Optionally, click **Delete branch** to delete your `welcome-workflow` branch.
-  Wait about 20 seconds, then refresh this page (the one you're following instructions from). Another workflow will run and will replace the contents of this README file with instructions for the next step.

# 5. Trigger the Workflow
---
The shell script in the workflow will run whenever a new pull request is opened.

**Seeing your _action_ in action**: The status of each workflow run that's triggered is shown in the pull request before it's merged: look for **All checks have passed** when you try out the steps below. You can also see a list of all the workflows that are running, or have finished running, in the **Actions** tab of your repository. From there, you can click on each workflow run to view more details and access log files.
![[Pasted image 20250501231404.png]]

## ⌨️ Activity: Trigger the workflow

-  Make a new branch named `test-workflow`.
-  Make a change, such as adding an emoji to your README.md file, and commit the change directly to your new branch.
-  In the **Pull requests** tab, create a pull request that will merge `test-workflow` into `main`.
-  Watch the workflow running in the checks section of the pull request.
-  Notice the comment that the workflow adds to the pull request.
- Wait about 20 seconds, then refresh this page (the one you're following instructions from). Another workflow will run and will replace the contents of this README file with instructions for the next step.

# Finish
---
Here's a recap of all the tasks you've accomplished in your repository:

- You've created your first GitHub Actions workflow file.
- You learned where to make your workflow file.
- You defined an event trigger, a job, and a step for your workflow.
- You're ready to automate anything you can dream of.

## What's next?

- Learn more about GitHub Actions by reading "[Learn GitHub Actions](https://docs.github.com/actions/learn-github-actions)"
- Use actions created by others in [awesome-actions](https://github.com/sdras/awesome-actions)
- We'd love to hear what you thought of this course [in our discussion board](https://github.com/orgs/skills/discussions/categories/hello-github-actions)
- [Take another course on GitHub Actions](https://skills.github.com/#automate-workflows-with-github-actions)
- Learn more about GitHub by reading the "[Get started](https://docs.github.com/get-started)" docs
- To find projects to contribute to, check out [GitHub Explore](https://github.com/explore)