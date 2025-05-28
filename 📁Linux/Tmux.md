[Tags::] #Beginner #tmux #red-hat

<br>

# Introduction
---
Tmux is a terminal multiplexer; it allows you to create several "pseudo terminals" from a single terminal. This is very useful for running multiple programs with a single connection, such as when you're remotely connecting to a machine using [Secure Shell (SSH)](https://www.redhat.com/sysadmin/access-remote-systems-ssh).

Tmux also decouples your programs from the main terminal, protecting them from accidentally disconnecting. You can detach tmux from the current terminal, and all your programs will continue to run safely in the background. Later, you can reattach tmux to the same or a different terminal.

In addition to its benefits with remote connections, tmux's speed and flexibility make it a fantastic tool to manage multiple terminals on your local machine, similar to a window manager. I've been using tmux on my laptops for over eight years. Some of tmux's features that help me and increase my productivity include:

- Fully customizable status bar
- Multiple window management
- Splitting window in several panes
- Automatic layouts
- Panel synchronization
- Scriptability, which allows me to create custom tmux sessions for different purposes

Here's an example of a customized tmux session:
![[tmux-custom-screen01.png.webp]]
Tmux offers some of the same functionality found in [Screen](https://www.gnu.org/software/screen/), which has been deprecated in some Linux distributions. Tmux has a more modern code base than Screen and offers additional customization capabilities.

# Install tmux
---
Tmux is available in the standard repositories with Fedora and [Red Hat Enterprise Linux (RHEL)](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), starting with RHEL 8. You can install it using DNF:

```bash
sudo dnf -y install tmux
```


_**[ Download now:**_ [_**A sysadmin's guide to Bash scripting**_](https://opensource.com/downloads/bash-scripting-ebook?intcmp=701f20000012ngPAAQ)_**. ]**_

# Get Started with tmux
---
To start using tmux, type `tmux` on your terminal. This command launches a tmux server, creates a default session (number 0) with a single window, and attaches to it.
```bash
tmux
```
![[Screenshot_20250502_145052.png]]
Now that you're connected to tmux, you can run any commands or programs as you normally would. For example, to simulate a long-running process:

```bash
c=1

while true; do echo "Hello $c"; let c=c+1; sleep 1; done
```
```
Hello 1
Hello 2
Hello 3
```

You can detach from your tmux session by pressing **Ctrl+B** then **D**(delay b/w ctrl+b and d). Tmux operates using a series of keybindings (keyboard shortcuts) triggered by pressing the "prefix" combination. By default, the prefix is **Ctrl+B**. After that, press **D** to detach from the current session.

```plaintext
[detached (from session 0)]
```

You're no longer attached to the session, but your long-running command executes safely in the background. You can list active tmux sessions with `tmux ls`:

```bash
tmux ls
```
```
0: 1 windows (created Sat Aug 27 20:54:58 2022)
```
_**[ Learn how to**_ [_**manage your Linux environment for success**_](https://www.redhat.com/en/engage/linux-management-ebook-s-201912231121)_**. ]**_

You can disconnect your SSH connection at this point, and the command will continue to run. When you're ready, reconnect to the server and reattach to the existing tmux session to resume where you left off:

```plaintext
$ tmux attach -t 0
Hello 72
Hello 73
Hello 74
Hello 75
Hello 76
^C  // Ctrl+C
```
All tmux commands can also be abbreviated, so, for example, you can enter `tmux a` , and it will work the same as `tmux attach`.

# Basic tmux keybindings
---
First, create a new tmux session if you're not already in one. You can name your session by passing the parameter `-s {name}` to the `tmux new` command when creating a new session:

```bash
tmux new -s Session1
```
![[Screenshot_20250502_150054.png]]
```bash
# list the tmux sessions
tmux ls 
```

| Key Binding           | Description                                                           |
| --------------------- | --------------------------------------------------------------------- |
| **Ctrl+B D**          | Detach from the current session                                       |
| **Ctrl+B %**          | Split the window into two panes horizontally                          |
| **Ctrl+B "**          | Split the window into two panes vertically                            |
| **Ctrl+B Arrow Keys** | Move Between panes                                                    |
| **Ctrl+B X**          | Close pane                                                            |
| **Ctrl+B C**          | Create a new window                                                   |
| **Ctrl+B N** or **P** | Move to the next or previous window                                   |
| **Ctrl+B 0(1,2...)**  | Move to a specific window by number                                   |
| **Ctrl+B :**          | Enter the command line to type commands. Tab completion is available. |
| **Ctrl+B ?**          | View all keybindings. Press **Q** to exit.                            |
| **Ctrl+B W**          | Open a panel to navigate across windows in multiple sessions.         |
| **Ctrl+B ,**          | Rename a pane                                                         |
| **Ctrl+B .**          | Move a pane to x position (0, 1, 2, .....)                            |
| **Ctrl+B S**          | Save the session (all)                                                |
| **Ctrl+B R**          | Restore the session (all)                                             |
| **Ctrl+B s**          | Switch between the sessions                                           | 

[Cheat Sheet for basic keybindings::[[#Cheatsheet]]]

# Configure tmux
---
By default, this file is located at `$HOME/.tmux.conf`.

For example, the default prefix key combination is **Ctrl+B**, but sometimes this combination is a little awkward to press, and it requires both hands. You can change it to something different by editing the configuration file. I like to set the prefix key to **Ctrl+A**. To do this, create a new configuration file and add these lines to it:

Open `$HOME/.tmux.conf` 

```bash
# Set the prefix to Ctrl+A
set -g prefix C-A

# Remove the old prefix
unbind C-B

# Send Ctrl+A to applications by pressing it twice
bind C-A send-prefix
```

# Customize the Status Bar
---
Tmux's status bar is fully customizable. You can change the colors of each section and what is displayed. There are so many options that it would require another article to cover them, so I'll start with the basics.

The standard green color for the entire status bar makes it difficult to see the different sections. It's particularly difficult to see how many windows you have open and which one is active.

[![tmux colors status bar](https://www.redhat.com/rhdc/managed-files/styles/wysiwyg_full_width/private/sysadmin/2022-09/tmux-colors-green01.png.webp?itok=12GrqZ36)](https://www.redhat.com/rhdc/managed-files/sysadmin/2022-09/tmux-colors-green01.png)

You can change that by updating the status bar colors. First, enter command mode by typing **Ctrl+B :** (or **Ctrl+A :** if you made the prefix configuration change above). Then change the colors with these commands:

- Change the status bar background color: `set -g status-bg cyan`
- Change inactive window color: `set -g window-status-style bg=yellow`
- Change active window color: `set -g window-status-current-style bg=red,fg=white`

Add these commands to your configuration file for permanent changes.

With this configuration in place, your status bar looks nicer, and it's much easier to see which window is active:

[![tmux colors](https://www.redhat.com/rhdc/managed-files/styles/wysiwyg_full_width/private/sysadmin/2022-09/tmux-colors01.png.webp?itok=3agvIGE0)](https://www.redhat.com/rhdc/managed-files/sysadmin/2022-09/tmux-colors01.png)

# What's next
---
Tmux is a fantastic tool to safeguard your remote connections and is useful when you spend a long time using the terminal. This article covers only the basic functionality, and there is much more to explore. For additional information about tmux, consult its official [wiki page](https://github.com/tmux/tmux/wiki).

You can also expand tmux's functionality with extra-official plugins. These plugins add more commands, integrate with applications such as [Vim](https://www.redhat.com/sysadmin/vim-power-commands), and add new functionality to the status bar. For more information, consult the [tmux plugins project](https://github.com/tmux-plugins/list).

# Cheatsheet
---
![[osdc_cheatsheet-tmux-2021.6.25.pdf]]