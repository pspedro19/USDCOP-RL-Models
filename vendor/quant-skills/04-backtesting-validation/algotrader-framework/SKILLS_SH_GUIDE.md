# How to Add AlgoTrader to Skills.sh

## What is Skills.sh?

[Skills.sh](https://skills.sh/) is the **open directory for AI agent skills**, launched by Vercel in January 2026. It's a centralized place to discover, browse, and track usage of skill packages that extend AI agent capabilities.

## ✅ Good News: Your Skill is Already Compatible!

Your AlgoTrader skill is **already properly structured** and ready for skills.sh! Here's what you have:

✅ **GitHub repository**: https://github.com/javajack/skill-algotrader
✅ **skill.json**: Proper manifest file with name, description, capabilities
✅ **Executable scripts**: run.sh, start.sh, algotrader.py
✅ **Documentation**: Comprehensive README, KNOWLEDGE, NUANCES
✅ **Templates**: Production-ready bot templates

## How Skills.sh Discovery Works

Skills.sh operates as an **open directory without formal review**:
- No submission or approval process required
- Skills are automatically discovered from GitHub repositories
- Users install directly from source using: `npx skills add owner/repo`

## Installation Command for Your Skill

Users can install your skill with:

```bash
npx skills add javajack/skill-algotrader
```

## Optional: Enhance Discoverability

While no submission is required, you can improve visibility:

### 1. Add GitHub Topics/Tags

Add these topics to your GitHub repository:

```bash
# Go to: https://github.com/javajack/skill-algotrader
# Click "⚙️ Settings" → Scroll to "Topics"
# Add these tags:
```

**Recommended topics:**
- `claude-code`
- `claude-skill`
- `ai-agent`
- `algorithmic-trading`
- `trading-bot`
- `zerodha`
- `python`
- `quantitative-trading`
- `skills-sh`

### 2. Update Repository Description

Go to your repo settings and add a clear description:

```
Quantitative trading skill for Indian equity markets with Zerodha integration.
Generate trading bots, fetch live index data, and manage risk with Claude Code.
```

### 3. Add Shields/Badges to README

Add these badges to the top of your README.md:

```markdown
[![Skills.sh](https://img.shields.io/badge/skills.sh-install-blue)](https://skills.sh/javajack/skill-algotrader)
[![GitHub](https://img.shields.io/github/stars/javajack/skill-algotrader?style=social)](https://github.com/javajack/skill-algotrader)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
```

### 4. Create a SKILL.md File (Optional Enhancement)

While you have `skill.json`, some ecosystems also look for `SKILL.md`. Create one:

```markdown
---
name: algotrader
description: Quantitative trading skill for Indian equity markets with Zerodha integration
author: Rakesh Waghela
version: 1.0.0
tags:
  - trading
  - zerodha
  - algorithmic-trading
  - python
  - claude-code
---

# AlgoTrader Skill

Comprehensive quantitative trading expert for Indian equity markets.

## Features

- 🎯 Interactive bot generation wizard
- 📊 Live index data fetcher (Nifty 50/100/Midcap/Smallcap)
- 🚀 Production-ready templates
- 📚 1,780 lines of trading knowledge
- ⚠️ 30+ production gotchas

## Installation

```bash
npx skills add javajack/skill-algotrader
```

## Usage

```bash
./start.sh wizard              # Generate trading bot
./start.sh universe            # Fetch market data
./start.sh my_bot              # Run bot
```

## Documentation

See [README.md](README.md) for comprehensive documentation.
```

## Check if Your Skill is Listed

After a few hours/days, check if your skill appears on:

1. **Skills.sh Directory**: https://skills.sh/
2. **Search for it**: https://skills.sh/?search=algotrader
3. **Direct link**: https://skills.sh/javajack/skill-algotrader

## Promote Your Skill

### Share Installation Command

Tell users to install with:

```bash
# One-line install
npx skills add javajack/skill-algotrader

# Or clone manually
git clone https://github.com/javajack/skill-algotrader.git ~/.claude/skills/algotrader
```

### Add to Claude Code Skills Path

Users can add your skill by:

```bash
# Option 1: Install via npx
npx skills add javajack/skill-algotrader

# Option 2: Clone to skills directory
cd ~/.claude/skills
git clone https://github.com/javajack/skill-algotrader.git algotrader

# Option 3: Custom skills path
export CLAUDE_SKILLS_PATH=~/work/skills
cd ~/work/skills
git clone https://github.com/javajack/skill-algotrader.git
```

### Share on Community Platforms

- **Twitter/X**: Share with #ClaudeCode #AITrading
- **Reddit**: r/ClaudeCode, r/algotrading
- **Hacker News**: Show HN post
- **Dev.to**: Write an article about your skill
- **LinkedIn**: Share with #AI #AlgorithmicTrading

## Leaderboard on Skills.sh

Skills.sh has a leaderboard showing most popular skills by install count. Promote your skill to climb the rankings!

**Current top skills:**
1. mcp-builder (anthropics/skills) - 8,975 installs
2. template-skill (anthropics/skills) - 5,400 installs
3. Your AlgoTrader skill - Coming soon! 🚀

## Maintaining Your Skill

Keep your skill updated to stay relevant:

```bash
# After making updates
git add .
git commit -m "Update: description of changes"
git push

# Tag releases
git tag -a v1.0.1 -m "Fix: folder scanner improvement"
git push --tags
```

## License

Add a LICENSE file (MIT recommended for open source):

```bash
# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Rakesh Waghela

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
EOF

git add LICENSE
git commit -m "Add MIT license"
git push
```

## Summary Checklist

- [x] ✅ GitHub repo created: javajack/skill-algotrader
- [x] ✅ Code pushed to GitHub
- [ ] ⏳ Add GitHub topics/tags
- [ ] ⏳ Update repository description
- [ ] ⏳ Add badges to README
- [ ] ⏳ Create SKILL.md (optional)
- [ ] ⏳ Add LICENSE file
- [ ] ⏳ Share on social media
- [ ] ⏳ Wait for skills.sh to index (automatic)

## Resources

- **Skills.sh**: https://skills.sh/
- **Your Skill**: https://github.com/javajack/skill-algotrader
- **Vercel Blog**: [Introducing skills](https://vercel.com/changelog/introducing-skills-the-open-agent-skills-ecosystem)
- **Claude Code Docs**: https://code.claude.com/docs/en/skills

---

**Your skill is ready! No submission needed - it's already discoverable! 🎉**

## Sources

- [Skills.sh - The Agent Skills Directory](https://skills.sh/)
- [Vercel: Introducing skills](https://vercel.com/changelog/introducing-skills-the-open-agent-skills-ecosystem)
- [Skills.sh Review (2026)](https://vibecoding.app/blog/skills-sh-review)
- [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills)
