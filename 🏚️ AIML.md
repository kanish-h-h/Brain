---
cssclass: dashboard
banner: "![[shutterstock_681234718.webp]]"
banner_y: 0.44
banner_x: 0.63637
---

# DashBoards


- **Crash Course (AIML):**
	- [[🏚️ AIML]]
	- [[Planning]]
	- [[Whole Plan]]
	- Canvas -> [[AIML Blueprint.canvas|AIML Blueprint]]


# Vault Info

- 🗄️ Recent file updates `$=dv.list(dv.pages('').sort(f=>f.file.mtime.ts,"desc").limit(4).file.link)`
<br>
- 🔖 Tagged: favorite `$=dv.list(dv.pages('#favourite').sort(f=>f.file.name,"desc").limit(4).file.link)`
<br>
- 〽️ Stats
    - File Count: `$=dv.pages().length`
    - Python Content: `$=dv.pages('"Python"').length`

