import transformers
import torch


# PROMPT = \
# "Rewrite the following paragraph into a tweet of no more than 280 characters. The final tweet needs to retain the main ideas of the text without altering its meaning. " + \
# "Maintain as much of the original style of the text as possible, including linguistic features such as rare words, prefixes, typos, and quantities. " + \
# "You can do so by replacing proper names and locations with Twitter handles, deleting unimportant information, and summarizing main ideas. \n" + \
# "Respond ONLY with the paraphrased text." + \
# "Original text: As vote counting continued on Tuesday, more Liberal MPs confirmed that Dutton was all but certain to lead the party, though a ballot will not be held until the count has concluded. " + \
# "NSW Liberal MP Sussan Ley was firming for the deputy's job. Victorian moderate senator Jane Hume and South Australian senator Anne Ruston were also being discussed.\n" + \
# "Tweet paraphrase: Vote counting cont'd. on Tuesday, with @MPDutton certain to lead the party. Ballot to be held after count concluded. NSW member @SussanLey firming for deputy's job, with mod. senator @JaneHume and @AnneRuston discussed.\n\n" + \
# "Original text: Don Giovanni is a metaphor not only for Trump, though, but also for how chaotic our media and our perception of the truth have become. " + \
# "Throughout the opera the characters are constantly switching identities, pulling on masks and pretending to be someone else, just as in our " + \
# "political life old identities don't seem to make sense anymore: what on earth is a 'conservative' these days, or 'the West'.\n" + \
# "Tweet paraphrase: Opera Don Giovanni is a metaphor not only for @DonaldTrump, but chaotic media + perception of truth. Thru it the chars switch " + \
# "identities, don masks and pretend, just as old identities don't make sense: what on earth is conversative/the West now?\n\n" + \
# "Original text:$TEXT$\nTweet paraphrase:"

# PROMPT = \
# "You are an expert at paraphrasing text documents into certain styles.\n\nYou will be rewriting paragraphs into tweets of NO MORE THAN 280 characters.\n\n" + \
# "# Guidelines\n\n1. The final tweet needs to retain the main ideas of the text without altering its meaning. " + \
# "Maintain as much of the original style of the text as possible, including linguistic features such as rare words, prefixes, typos, and quantities. " + \
# "You can do so by replacing proper names and locations with Twitter handles, deleting unimportant information, and summarizing main ideas. \n" + \
# "Respond ONLY with the paraphrased text." + \
# "Original text: As vote counting continued on Tuesday, more Liberal MPs confirmed that Dutton was all but certain to lead the party, though a ballot will not be held until the count has concluded. " + \
# "NSW Liberal MP Sussan Ley was firming for the deputy's job. Victorian moderate senator Jane Hume and South Australian senator Anne Ruston were also being discussed.\n" + \
# "Tweet paraphrase: Vote counting cont'd. on Tuesday, with @MPDutton certain to lead the party. Ballot to be held after count concluded. NSW member @SussanLey firming for deputy's job, with mod. senator @JaneHume and @AnneRuston discussed.\n\n" + \
# "Original text: Don Giovanni is a metaphor not only for Trump, though, but also for how chaotic our media and our perception of the truth have become. " + \
# "Throughout the opera the characters are constantly switching identities, pulling on masks and pretending to be someone else, just as in our " + \
# "political life old identities don't seem to make sense anymore: what on earth is a 'conservative' these days, or 'the West'.\n" + \
# "Tweet paraphrase: Opera Don Giovanni is a metaphor not only for @DonaldTrump, but chaotic media + perception of truth. Thru it the chars switch " + \
# "identities, don masks and pretend, just as old identities don't make sense: what on earth is conversative/the West now?\n\n" + \
# "Original text:$TEXT$\nTweet paraphrase:"

PROMPT = '\n'.join(open('src/data_preprocessing/paraphrasing/prompt.txt', 'r').readlines())

MODEL = None

def construct_prompt(text):
    return PROMPT.replace('$TEXT$', text)

def paraphrase(text, model_id):
    global MODEL
    if MODEL is None:
        MODEL = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    prompt = construct_prompt(text)
    message = [{'role': 'user', 'content': prompt}]
    terminators = [
            MODEL.tokenizer.eos_token_id,
            MODEL.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    output = MODEL(
            message,
            max_new_tokens=512,
            eos_token_id=terminators,
            pad_token_id=MODEL.tokenizer.eos_token_id,
            do_sample=True,
        )
    
    return output[0]['generated_text'][-1]['content']