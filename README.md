# SD.Next Agent Scheduler

Fork of AgentScheduler, intend to keep SD.Next compatibility 

## Table of Content

## Compatibility

This version of AgentScheduler is might be compatible with:

- SD.Next: https://github.com/vladmandic/sdnext
- SD.Next ipex fork (for iGPU) https://github.com/liutyi/sdnext/tree/ipex

## Installation

### Using with SD.Next

- The extension is already included in
  - https://github.com/liutyi/sdnext/tree/ipex
  - https://github.com/liutyi/sdnext/tree/pytorch
- Need Manual install in SD.Next since about 2025-10 (no longer built-in)
  - https://github.com/vladmandic/sdnext

### Using the built-in extension list

1. Open the Extensions tab
2. Open the "Manual Install" sub-tab
3. Paste the repo url: https://github.com/liutyi/sdnext-scheduler.git
4. Click "Install"

![sdnext-scheduler-manual-install](https://github.com/user-attachments/assets/9acf4a20-2ac0-4752-aa3a-f7ad2b6fdab8)


### Manual clone

```bash
git clone "https://github.com/liutyi/sdnext-scheduler.git" extensions/sdnext-scheduler
```

(The second argument specifies the name of the folder, you can choose whatever you like).

## Basic Features


1️⃣ Input your usual Prompts & Settings. **Enqueue** to send your current prompts, settings, controlnets to **AgentScheduler**.

![sdnext-schedule](https://github.com/user-attachments/assets/4bb9e388-27c4-4342-b7cc-de2d09269400)

2️⃣ **AgentScheduler** Extension Tab.

![sdnext-schedule-task-list](https://github.com/user-attachments/assets/3b71f0e4-21a1-4140-a2b2-b43296a7cc9a)


3️⃣ See all queued tasks, current image being generated and tasks' associated information. **Drag and drop** the handle in the begining of each row to reaggrange the generation order.

4️⃣ **Pause** to stop queue auto generation. **Resume** to start.

5️⃣ Press ▶️ to prioritize selected task, or to start a single task when queue is paused. **Delete** tasks that you no longer want.

6️⃣ Show queue history.

![sdnext-schedule-history](https://github.com/user-attachments/assets/df04356d-fca5-4170-81f9-e7f0d1b8436b)


7️⃣ **Filter** task status or search by text.

![sdnext-shedule-search](https://github.com/user-attachments/assets/ac28e155-0091-4a8d-8dff-660201475f26)


8️⃣ **Bookmark** task to easier filtering.

![sdnext-schedule-bookmark](https://github.com/user-attachments/assets/d07900d5-2fa3-4410-b9c2-d377f8c75b82)

9️⃣ Double click the task id to **rename** and quickly update basic parameters. Click ↩️ to **Requeue** old task.

![sdnext-schedule-rename](https://github.com/user-attachments/assets/cd079f14-eeed-4dc0-aead-ce1c28a89341) 

🔟 Click on each task to **view** the generation results.


## Hidden Features:

#### Queue all checkpoints at the same time

Right click the `Enqueue` button and select `Queue with all checkpoints` to quickly queue the current setting with all available checkpoints.


#### Queue with a subset of checkpoints
![sdnext-schedule-all-checkpoints](https://github.com/user-attachments/assets/4627a936-8faf-4d69-921a-16afbc5a0a81)


With the custom checkpoint select enabled (see [Extension Settings](#extension-settings) section below), you can select a folder (or subfolder) to queue task with all checkpoints inside. Eg: Select `anime` will queue `anime\AOM3A1B_oragemixs`, `anime\counterfeit\Counterfeit-V2.5_fp16` and `anime\counterfeit\Counterfeit-V2.5_pruned`.

#### Edit queued task

Double click a queued task to edit. You can name a task by changing `task_id` or update some basic parameters: `prompt`, `negative prompt`, `sampler`, `checkpoint`, `steps`, `cfg scale`.


## Extension Settings

Go to `Settings > Agent Scheduler` to access extension settings.

**Disable Queue Auto-Processing**: Check this option to disable queue auto-processing on start-up. You can also temporarily pause or resume the queue from the Extension tab.

**Queue Button Placement**: Change the placement of the queue button on the UI.

**Hide the Checkpoint Dropdown**: The Extension provides a custom checkpoint dropdown.


By default, queued tasks use the currently loaded checkpoint. However, changing the system checkpoint requires some time to load the checkpoint into memory, and you also cannot change the checkpoint during image generation. You can use this dropdown to quickly queue a task with a custom checkpoint.

**Auto Delete Queue History**: Select a timeframe to keep your queue history. Tasks that are older than the configured value will be automatically deleted. Please note that bookmarked tasks will not be deleted.

## API Access

All the functionality of this extension can be accessed through HTTP APIs. You can access the API documentation via `http://127.0.0.1:7860/docs`. Remember to include `--api` in your startup arguments.


#### Queue Task

The two apis `/agent-scheduler/v1/queue/txt2img` and `/agent-scheduler/v1/queue/img2img` support all the parameters of the original webui apis. These apis response the task id, which can be used to perform updates later.

```json
{
  "task_id": "string"
}
```

#### Download Results

Use api `/agent-scheduler/v1/results/{id}` to get the generated images. The api supports two response format:

- json with base64 encoded

```json
{
  "success": true,
  "data": [
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAAN...",
      "infotext": "1girl\nNegative prompt: EasyNegative, badhandv4..."
    },
    {
      "image": "data:image/png;base64,iVBORw0KGgoAAAAN...",
      "infotext": "1girl\nNegative prompt: EasyNegative, badhandv4..."
    }
  ]
}
```

- zip file with querystring `zip=true`

#### API Callback

Queue task with param `callback_url` to register an API callback. Eg:

```json
{
  "prompt": "1girl",
  "negative_prompt": "easynegative",
  "callback_url": "http://somehost:port/task_completed"
}
```

The callback endpoint must support `POST` method with body in `multipart/form-data` encoding. Body format:

```json
{
  "task_id": "abc123",
  "status": "done",
  "files": [list of image files],
}
```

Example code of the endpoint handle with `FastApi`:

```python
from fastapi import FastAPI, UploadFile, File, Form

@app.post("/task_completed")
async def handle_task_completed(
    task_id: Annotated[str, Form()],
    status: Annotated[str, Form()],
    files: Optional[List[UploadFile]] = File(None),
):
    print(f"Received {len(files)} files for task {task_id} with status {status}")
    for file in files:
        print(f"* {file.filename} {file.content_type} {file.size}")
        # ... do something with the file contents ...

# Received 1 files for task 3cf8b150-f260-4489-b6e8-d86ed8a564ca with status done
# * 00008-3322209480.png image/png 416400
```

## Troubleshooting

Make sure that you are running the latest version of the extension and an updated version of the WebUI.

- To update the extension, go to `Extension` tab and click `Check for Updates`, then click `Apply and restart UI`.
- To update the WebUI it self, you run the command `git pull origin master` in the same folder as webui.bat (or webui.sh).

Steps to try to find the cause of issues:

- Check the for errors in the WebUI output console.
- Press F12 in the browser then go to the console tab and reload the page, find any error message here.

Common errors:

**TypeError: issubclass() arg 1 must be a class**
Please update the extension, there's a chance it's already fixed.

**TypeError: Object of type X is not JSON serializable**
Please update the extension, it should be fixed already. If not, please fire an issue report with the list of installed extensions.

## License

This project is licensed under the Apache License 2.0.

## Disclaimer

The author(s) of this project are not responsible for any damages or legal issues arising from the use of this software. Users are solely responsible for ensuring that they comply with any applicable laws and regulations when using this software and assume all risks associated with its use. The author(s) are not responsible for any copyright violations or legal issues arising from the use of input or output content.

---

## Initially crafted by the people building
- [SIPHER//AGI](https://sipheragi.com), 
- [PROTOGAIA](https://protogaia.com), 
- [ATHERLABS](https://atherlabs.com/),
- [SIPHER ODYSSEY](http://playsipher.com/)
