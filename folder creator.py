# import os

# for i in range(1, 101):
#     if not os.path.exists(i):
#         os.makedirs(str(i))

# !pip install transformers

from transformers import pipeline

data = """
Retailers, the makers of foods marketed for weight loss, and other types of companies could see knock-on effects from the rise of diabetes and weight loss drugs like Ozempic.

As they do every summer, publicly traded companies posted their second-quarter results while Americans were baring their bodies on the beach. But this year, the timing was apt. On several earnings calls in August, chief executives reassured investors that the Ozempic revolution had not left them in the dust, and that they could somehow share in the blazing success of new diabetes and weight loss drugs. “It puts us in a good position to be a solution for those who are on the drugs,” said Dan R. Chard, the chief executive of Medifast, which makes diet products like shakes and protein bars, adding: “They’re looking for guidance.” He told analysts this even while explaining that new-generation drugs had helped pummel earnings, down 34.7 percent year on year. “We will continue to study this,” Michael Johnson, the chief executive of the nutritional supplement maker Herbalife, told investors. “And when we see an opportunity to capitalize on it, we will.” In theory, that opportunity — both for making profits and for losing fortunes — could be vast not only for the companies behind these drugs but also for some in completely different industries.

Known as GLP-1 drugs, the medications are already driving big profits. Novo Nordisk makes both Ozempic, which has been approved only for Type 2 diabetes, and its close relative Wegovy, which has been approved for weight loss. They mimic a glucagon-like peptide that regulates appetite in the brain, leaving people feeling sated for hours. Together, they helped send Novo’s earnings rocketing up 32 percent in the first half of this year, and Novo’s market value is now larger than the entire Danish economy. Eli Lilly’s sales surged 28 percent in the second quarter, thanks to another diabetes drug, Mounjaro, which the Food and Drug Administration may approve for weight loss this year. And the full potential isn’t even clear yet. The market for weight loss drugs is huge: There are roughly 750 million obese people worldwide, including about 42 percent of adults in the United States, where obesity-related illnesses incur billions of dollars in health care costs each year. But Novo says GLP-1 drugs could eventually have other uses, like helping prevent cardiovascular disease among obese adults. There are signs they could treat addiction and even Alzheimer’s, too. “The market potential is very, very significant,” Novo’s chief financial officer, Karsten Knudsen, told me when I visited the company in June. “We’re operating in kind of unusual territory.”

The ripple effects are widening. Retailers like Walmart, Kroger and Rite Aid say GLP-1 prescriptions are bringing more people into stores, where they make other purchases. Walmart’s chief executive, Doug McMillon, told analysts in August that its executives “expect consumables, and health and wellness, primarily due to the popularity of some GLP-1 drugs, to grow as a percent of total.” Medtronic’s chief executive, Geoff Martha, said the company had seen a “modest” dip in bariatric surgery, presumably as people opted for weight loss drugs instead. And some analysts believe the drugs could disrupt the American diet. “If you’re eating fast food every day, you’ll probably continue to eat fast food every day,” James van Geelen of Citrinas Capital Management said on Bloomberg’s “Odd Lots” podcast. “You will just eat a lot less of it.” Still, there is room for other approaches to fighting obesity. “These drugs are game changers, but with an asterisk,” said David Ludwig, an obesity specialist and pediatrics professor at Harvard Medical School. (The drugs come with a long list of side effects.) “Even if you can reduce weight across the population with drugs, it’s not going to eliminate the risks of a poor diet.” Flush with cash, Novo agrees. “We need to be looking at what’s the next thing,” its executive vice president for commercial strategy Camilla Sylvest, told me. In June, the company launched an obesity prevention unit near Copenhagen, to research how to stop the disease before people need to take drugs to lose weight. — Vivienne Walt

The U.S. labor market begins to look like its old self. Employers added 187,000 jobs in August, the Labor Department reported Friday, and unemployment rose to 3.8 percent as the economy continued to lose momentum built up after pandemic lockdowns. Commerce Secretary Gina Raimondo visits China. She had the tricky task of promoting trade between the two superpowers while holding firm on technology export limits imposed in the name of American national security. The two countries agreed to create new dialogues, including a working group for commercial issues. The White House names the first drugs set for Medicare price negotiations. The long-awaited list of 10 medicines will be subject to a landmark new program meant to reduce costs for Medicare. Drugmakers have pushed back against the plan, including in court, and Republicans have criticized the initiative as government overreach. UBS reports a $29 billion quarterly profit, with an asterisk. The huge gain — the biggest in banking history — stems from the bank’s acquisition of its rival Credit Suisse this spring for about $3.2 billion, a steep discount that is skewing UBS’s results. But it belies the challenges that UBS faces as it moves to complete the largest takeover of a bank since the 2008 financial crisis. When Emily Weiss stepped down last year as the chief executive of Glossier, the skin care and beauty brand she founded in 2014, some called it the end of the “girlboss.” That archetype — of media-savvy female founders with venture-darling, millennial-focused start-ups — had been propelled by “#Girlboss,” the Nasty Girl founder Sophia Amoruso’s 2014 memoir.

Glossier, with its direct-to-consumer model and voice-y website, changed the way women buy makeup, eventually passing a $1 billion valuation. But the brand stumbled as it struggled to move into brick-and-mortar retail; faced criticism from retail employees who alleged a toxic, racist working environment; and shelved projects like a makeup line that departed from its dewy, barely-there look. DealBook spoke to Marisa Meltzer, author of the upcoming book “Glossy: Ambition, Beauty, and the Inside Story of Emily Weiss’s Glossier,” about what lessons we might glean from Weiss and the #Girlboss movement. This interview has been edited and condensed for clarity. Can you contextualize the #Girlboss movement? It was pretty offensive and diminutive. No one except for Sophia was calling themselves out as a girlboss. But it was also something that benefited them because it attracted interest. It was a way for them to get press about their businesses that wasn’t the typical things that female founders and C.E.O.s sometimes had to do, like a fashion spread.

There was a big debate at the time over whether the press would have covered the scandals at companies like Outdoor Voices, Man Repeller and Glossier differently if they had men at the helm. What do you think? I think there was a bit of a thirst for blood. These women had been propped up in a way that was kind of annoying — I’m sure it was annoying to them, too. Some of those companies had real problems, like being sued over firing pregnant employees. And other companies had, like Glossier, an accusation of having a workplace, largely for their retail employees, that wasn’t ideal. That’s different than criminal behavior. The reality is these companies weren’t the same. The women at their helms were not all the same. And they weren’t making the same mistakes. And they also didn’t have the same level of success. What happens to Glossier now? Glossier seems to have taken the time since Emily stepped down to re-evaluate. They decided to really belatedly go into retail. They launched in Sephora last February. The larger task that they’re trying to do is make the company in better shape for an exit. 
"""

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-xlarge-mnli")

def check_model(data):
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )
    print('Fill blank: ')
    print(fill_mask(f"{data} {fill_mask.tokenizer.mask_token}."))

    # print('Fill blank: ')
    # print(fill_mask(f" {fill_mask.tokenizer.mask_token}."))

print('Check model ...')
check_model(data)

# def sentiment_using_bert(docs):
#     try:
#         # Truncate or split the input data to fit within the model's sequence length
#         max_sequence_length = 512
#         truncated_data = docs[:max_sequence_length]

#         # Initialize the BERT pipeline
#         pipe = pipeline("fill-mask", model="microsoft/deberta-xlarge-mnli")

#         # Get sentiment score for the truncated data
#         sentiment_score = pipe(truncated_data)

#         print(sentiment_score)
#         print("Sentiment : ", sentiment_score[0]['label'])

#         if sentiment_score[0]['label'] == "ENTAILMENT":
#             return 'POS'
#         elif sentiment_score[0]['label'] == "CONTRADICTION":
#             return 'NEG'
#         else:
#             return 'NEU'

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         return 'NEU'

# data_content = data["hits"]["hits"][0]["_source"]["content"]
# sentiment_using_bert(data)
