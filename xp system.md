## **XP earning per action:**
Each game spin (or other action) grants some XP. For slot games, a typical model is to award **base XP per spin** plus bonuses for outcomes. 
For example, you might give `10 XP` for every spin regardless of result, so the player always makes progress. Wins then multiply that. As noted by Slotomania’s guide, _“Experience Points (XP) are the points you can earn for every spin on a Slotomania slot game. As you gain more XP, you can increase your Player Level”_​ [pokernews.com](https://www.pokernews.com/free-online-games/slotomania/get-free-coins.htm#:~:text=Experience%20Points%20). 
In practice, many casinos tie XP to betting: “players accumulate points based on their betting activity” ​[smartico.ai](https://smartico.ai/blog-post/gamification-tactics-high-roller-online-casinos#:~:text=experience%20for%20high%20rollers,for%20exclusive%20rewards%20such%20as). 

## **Schema**

- **Base XP:** A fixed award (e.g. 5–20 XP) per spin, so even a losing spin gives something.
    
- **Bet multiplier:** Scale XP by the bet size. E.g. `XP += floor(Bet × α)`, so larger wagers give more XP. This mimics how Slotomania unlocks higher bets at higher levels [pokernews.com](https://www.pokernews.com/free-online-games/slotomania/get-free-coins.htm#:~:text=However%2C%20it%20should%20be%20noted,comprehensive%20array%20of%20slot%20games).
    
- **Win bonus:** Additional XP for winning. For instance, `XP += floor(WinAmount / β)` (scaled by a factor β). Or use tiers: a “big win” multiplier (5×, 10× the bet) grants 2× or 3× the base XP, a “mega win" grants 5× or more.
    
- **Special events:** Award extra XP for hitting bonus rounds, free spins, or rare symbols. For example, triggering a bonus wheel might give a one-time bonus of 100 XP.
    

## **Formula example:** 
One might combine these as

>$XP_{gain} = Base_{XP} + floor(bet_{amount} * M) + floor(win_{amount} * W)$

where `M` and `W` are scaling factors. For instance, set 
$Base_{XP}$=10,
M=0.1,
W=0.2
(so betting 100 coins gives 10 XP plus extra on wins). A concrete example:

|Bet|Win|Mult.|XP gained (example)|
|---|---|---|---|
|100|0|0×|10 (base)|
|100|50|0.5×|10 + floor(100_0.1)+floor(50_0.2) = 10+10+10 = 30|
|100|500|5×|10+10+100 = 120|
|100|1000|10×|10+10+200 = 220|

In this toy model, 
- a small win (50 on a 100 bet) gives modest extra XP (30)
- while a big win (500) jumps to 120 XP
- and a jackpot (1000) to 220 XP.
- The exact formula can vary, but the principle is the same: **bigger bets and bigger wins yield more XP**. This motivates players to gamble higher within their means.

## **Bet size and multipliers:**
To incorporate win multipliers, you could also tie XP directly to the multiplier. For instance:

```python
if win_multiplier < 5:
	XP += floor(bet * 1)
else if win_multiplier < 10:
	XP += floor(bet * 3)
else:
	XP += floor(bet * 5)
```

This gives 1×, 3×, or 5× XP per unit bet depending on win size. 

Another approach is $XP += floor(bet * win_multiplier / 10)$, so a
- 10× win yields 10× the bet in XP (plus base).
- There’s flexibility here.
- The key is to keep progression balanced: ensure very big wins don’t grant _excessively_ huge XP that skip levels, but do feel substantially more rewarding.

## **Level progression curve:** 
Define how much total XP is needed for each level. It should grow so early levels are easy to hit (to hook the player) and later levels require more play (to sustain engagement).
A common practice is an **increasing curve**, either linear-progression or exponential​      ​[gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Linear%20progression%20curve).


Examples:
- _Linear+progressive:_ Add a growing flat amount each level. E.g. Level 2 needs 100 XP, Level 3 needs 200 more (total 300), Level 4 needs 300 more (total 600), etc. This “easy early, steady growth” approach prevents values from exploding ​[gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Linear%20progression%20curve).
    
- _Exponential:_ Multiply the requirement by a factor each time. For example, “next level XP = previous level XP × 1.4” ​[gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Exponential%20curve). Starting at 100, this yields 100, 140, 196, 274… XP for levels 2,3,4,5. Early gains are quick, but high levels require large XP.
    

A sample level table using a simple increasing increment rule might be:

|Level|Total XP required (example)|
|---|---|
|1|0|
|2|100|
|3|300|
|4|600|
|5|1000|
|6|1500|
|7|2100|

Here each level takes 100 more XP than the last (i.e. +100, +200, +300, +400…). This is a **linear-progression curve** giving fairly quick early levels but gradually longer play for high levels​.

Alternatively, an exponential rule like could be used for a steeper climb.
>$XP_{next} = 100 * (1.3)^{(level-2)}$

 Designers often fine-tune this by hand to hit desired playtime. The cited industry guideline notes that exponential curves _“offer easy early thresholds and much more demanding later ones”_ ​[gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Exponential%20curve), which can be effective if balanced correctly.

## **Level-up rewards:** 
Each time the player reaches a new level, grant meaningful bonuses. For example:

- **Coin bonus:** Award a lump-sum of in-game currency. Many games do this (Slotomania gives free coins on leveling) ​[betting.net](https://www.betting.net/reviews/slotomania/promo-code/#:~:text=Playing%20games%20earns%20you%20experience,faster%20and%20reap%20these%20rewards). Even a small bonus (e.g. +10–25% of the daily stake) feels satisfying.
    
- **Unlock new features:** Raise betting limits or unlock new games, tables, or slot titles. Slotomania explicitly ties higher levels to _“start betting larger amounts”_ and accessing _“more comprehensive array of slot games”_ ​[pokernews.com](https://www.pokernews.com/free-online-games/slotomania/get-free-coins.htm#:~:text=However%2C%20it%20should%20be%20noted,comprehensive%20array%20of%20slot%20games). This creates a sense of progression (you feel “stronger” as you level up).
    
- **Bonus spins/tokens:** Grant extra free spins, bonus game triggers, or special event entries on level-up. This directly ties into the spin-wheel mechanic itself.
    
- **VIP-style perks:** For high levels, offer VIP treatment: for example, `Stake.com’s` `VIP` program gives _“level-up bonuses”_ (cash bonuses based on wagering) whenever the player hits a new tier​[sportsgambler.com](https://www.sportsgambler.com/review/stake/vip-bonus/#:~:text=Level%20up%20bonuses). While our game may not need a full rakeback system, we can mimic it with one-time bonuses or boosted earnings.
    
- **Badges and titles:** Award cosmetic badges, titles, or profile items. These don’t affect gameplay but leverage social status.
    
- **Consistency/Mission boosts:** Unlock double-XP days or special missions for reaching certain levels.
    

In all cases, make the benefits clear. A typical implementation (as seen in social slots) is: _“You leveled up! Here are 1,000 bonus coins plus access to Mega Slots!”_ This positive feedback loop encourages more play.

## **Summary of XP system best practices:** 
Tie XP closely to core actions (spins, bets), scale with effort (bet size, big wins), and use a smooth but accelerating curve for levels ​[gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Exponential%20curve)​ [gamedeveloper.com](https://www.gamedeveloper.com/design/quantitative-design---how-to-define-xp-thresholds-#:~:text=Linear%20progression%20curve).

Reward each level-up visibly (coin bonus, unlocks) so players feel rewarded for their time​ [betting.net](https://www.betting.net/reviews/slotomania/promo-code/#:~:text=Playing%20games%20earns%20you%20experience,faster%20and%20reap%20these%20rewards) ​[pokernews.com](https://www.pokernews.com/free-online-games/slotomania/get-free-coins.htm#:~:text=However%2C%20it%20should%20be%20noted,comprehensive%20array%20of%20slot%20games). 

In a well-known example, `Slotomania` players are explicitly told _“the more you spin, the more XP you earn, pushing you higher up the VIP ladder where you’ll unlock exclusive rewards”​_ [pokernews.com](https://www.pokernews.com/free-online-games/slotomania/get-free-coins.htm#:~:text=However%2C%20it%20should%20be%20noted,comprehensive%20array%20of%20slot%20games)​ [betting.net](https://www.betting.net/reviews/slotomania/promo-code/#:~:text=Playing%20games%20earns%20you%20experience,faster%20and%20reap%20these%20rewards). 

`Stake.com` similarly emphasizes earning loyalty points via wagers to climb VIP tiers and gain bonuses​ [sportsgambler.com](https://www.sportsgambler.com/review/stake/vip-bonus/#:~:text=casinos%2C%20Stake,points%20to%20improve%20their%20status)​  [sportsgambler.com](https://www.sportsgambler.com/review/stake/vip-bonus/#:~:text=Level%20up%20bonuses). 