
```
{
	"id" = "game name"
	
	"matrix_size": {
		"x": row,
		"y": column
		},
	
	"linesApiData": [
	[ a, a, a, a, a,]  # lines -> 20
	[.......................]
	[.......................]
	],
	
	"linesCount": [1,5,15,20], # lines user selection
	
	"bets": [] # no. of bets
	
	"bonus": {
		"type": "layerTap",
		"isEnabled": true,
		"noOfItem": 3,
		"payout": [[], [], []], # three reels
		"payOutProb": [[1,1,1,1], [1,1,1], [1,1]]
	},
	
	"gamble": {
		"type": "gamble type",
		"isEnabled": true
	},
	
	"Symbols":[
		{
			"Name":"Symbol Name",
			"Id": symbol_id,
			"reelInstance":{
				"reel 0": frequency,
				"reel 1": frequency,
				"reel 2": frequency,
				"reel 3": frequency,
				"reel 4": frequency,
			},
			"useWildSub":true,  # subsitution of wild symbol
			"multiplier":[
				[payout, spins],
				[payout, spins],
				[payout, spins]
			]
		},
		
		{
			"Name":"Jackpot",
            "Id":9,
            "reelInstance":{"reels":frequency},
            "useWildSub":false,
            "symbolCount":5,
            "pay":5000
		},
		
		{
			"Name":"Bonus",
            "Id":10,
            "reelInstance":{"reels":frequency},
            "useWildSub":false,
            "symbolCount":3,
		},
		
		{
			"Name":"wild",
            "Id":11,
            "reelInstance":{"reels":frequency},
            "useWildSub":false,
            "multiplier":[
            
            ]
		}
	]
}
```
