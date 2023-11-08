import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline

# from transformers import AutoTokenizer, AutoModelForCausalLM

hf_model = "/cluster/work/lawecon/Work/raj/llama2models/13b-chat-hf"  # @param ["meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]
temperatures = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def print_function_name(func):
    def wrapper(*args, **kwargs):
        print(f"Using function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

print("######")
print(hf_model + " || generate")
print("######")

# TODO
# make the matrix to find a list of good sys and user message prompts and on them run the temperature tests.
# TODO
# print(all the parameters of generation)
# TODO
# print(experiment with temperature to find the best temperature for this task.)
# TODO
# start the semantic similarity experiment

# Facebook Defaults
# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.8,
#     top_p: float = 0.95,
#     max_seq_len: int = 512,
#     max_batch_size: int = 32,
# ):
tokenizer = LlamaTokenizer.from_pretrained(
    hf_model,
    # use_auth_token=True,
)

model = LlamaForCausalLM.from_pretrained(
    hf_model,
    device_map="auto",
    torch_dtype=torch.float16,
    #  use_auth_token=True,
    load_in_8bit=False,
    load_in_4bit=False,
    rope_scaling={"type": "dynamic", "factor": 2.0},
    # max_memory={0:"40.0GiB", 1:"40.0GiB", 2:"40.0GiB", 3:"40.0GiB"},
    max_memory={0:"40.0GiB", 1:"40.0GiB"},
    do_sample=True, #necessary to make temperature work
    # temperature=0.7 # ?? failed
)
# print(model.generation_config)
# print("######")
# Use a pipeline for later
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=4096,
                return_full_text=False,
                # temperature=0.4 # ??
                )
# print(pipeline)
# print("######")
# @print_function_name
def generate(prompt):
    with torch.autocast("cuda", dtype=torch.float16):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            # temperature = temp #??
        )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        #       generated_text = final_outputs[len(prompt):]
        # final_outputs = cut_off_text(final_outputs, '</s>')
        print(final_outputs)
        # torch.cuda.empty_cache() 
        print("#####################################################")

# @print_function_name
def pipe_generate(prompt):
    sequences = pipe(prompt)
    for seq in sequences:
        print(seq['generated_text'])
    # torch.cuda.empty_cache() 
    print("#####################################################")

story_1 = """The woman had died without pain, quietly, as a woman should whose life had been blameless. Now she was resting in her bed, lying on her back, her eyes closed, her features calm, her long white hair carefully arranged as though she had done it up ten minutes before dying. The whole pale countenance of the dead woman was so collected, so calm, so resigned that one could feel what a sweet soul had lived in that body, what a quiet existence this old soul had led, how easy and pure the death of this parent had been.
Kneeling beside the bed, her son, a magistrate with inflexible principles, and her daughter, Marguerite, known as Sister Eulalie, were weeping as though their hearts would break. She had, from childhood up, armed them with a strict moral code, teaching them religion, without weakness, and duty, without compromise. He, the man, had become a judge and handled the law as a weapon with which he smote the weak ones without pity. She, the girl, influenced by the virtue which had bathed her in this austere family, had become the bride of the Church through her loathing for man.
They had hardly known their father, knowing only that he had made their mother most unhappy, without being told any other details.
The nun was wildly-kissing the dead woman's hand, an ivory hand as white as the large crucifix lying across the bed. On the other side of the long body the other hand seemed still to be holding the sheet in the death grasp; and the sheet had preserved the little creases as a memory of those last movements which precede eternal immobility.
A few light taps on the door caused the two sobbing heads to look up, and the priest, who had just come from dinner, returned. He was red and out of breath from his interrupted digestion, for he had made himself a strong mixture of coffee and brandy in order to combat the fatigue of the last few nights and of the wake which was beginning.
He looked sad, with that assumed sadness of the priest for whom death is a bread winner. He crossed himself and approaching with his professional gesture: "Well, my poor children! I have come to help you pass these last sad hours." But Sister Eulalie suddenly arose. "Thank you, "father, but my brother and I prefer to remain alone with her. This is our last chance to see her, and we wish to be together, all three of us, as we--we--used to be when we were small and our poor mo--mother----"
Grief and tears stopped her; she could not continue.
Once more serene, the priest bowed, thinking of his bed. "As you wish, my children." He kneeled, crossed himself, prayed, arose and went out quietly, murmuring: "She was a saint!"
They remained alone, the dead woman and her children. The ticking of the clock, hidden in the shadow, could be heard distinctly, and through the open window drifted in the sweet smell of hay and of woods, together with the soft moonlight. No other noise could be heard over the land except the occasional croaking of the frog or the chirping of some belated insect. An infinite peace, a divine melancholy, a silent serenity surrounded this dead woman, seemed to be breathed out from her and to appease nature itself.
Then the judge, still kneeling, his head buried in the bed clothes, cried in a voice altered by grief and deadened by the sheets and blankets: "Mamma, mamma, mamma!" And his sister, frantically striking her forehead against the woodwork, convulsed, twitching and trembling as in an epileptic fit, moaned: "Jesus, Jesus, mamma, Jesus!" And both of them, shaken by a storm of grief, gasped and choked.
The crisis slowly calmed down and they began to weep quietly, just as on the sea when a calm follows a squall.
A rather long time passed and they arose and looked at their dead. And the memories, those distant memories, yesterday so dear, to-day so torturing, came to their minds with all the little forgotten details, those little intimate familiar details which bring back to life the one who has left. They recalled to each other circumstances, words, smiles, intonations of the mother who was no longer to speak to them. They saw her again happy and calm. They remembered things which she had said, and a little motion of the hand, like beating time, which she often used when emphasizing something important.
And they loved her as they never had loved her before. They measured the depth of their grief, and thus they discovered how lonely they would find themselves.
It was their prop, their guide, their whole youth, all the best part of their lives which was disappearing. It was their bond with life, their mother, their mamma, the connecting link with their forefathers which they would thenceforth miss. They now became solitary, lonely beings; they could no longer look back.
The nun said to her brother: "You remember how mamma used always to read her old letters; they are all there in that drawer. Let us, in turn, read them; let us live her whole life through tonight beside her! It would be like a road to the cross, like making the acquaintance of her mother, of our grandparents, whom we never knew, but whose letters are there and of whom she so often spoke, do you remember?"
Out of the drawer they took about ten little packages of yellow paper, tied with care and arranged one beside the other. They threw these relics on the bed and chose one of them on which the word "Father" was written. They opened and read it.
It was one of those old-fashioned letters which one finds in old family desk drawers, those epistles which smell of another century. The first one started: "My dear," another one: "My beautiful little girl," others: "My dear child," or: "My dear daughter." And suddenly the nun began to read aloud, to read over to the dead woman her whole history, all her tender memories. The judge, resting his elbow on the bed, was listening with his eyes fastened on his mother. The motionless body seemed happy.
Sister Eulalie, interrupting herself, said suddenly:
"These ought to be put in the grave with her; they ought to be used as a shroud and she ought to be buried in it." She took another package, on which no name was written. She began to read in a firm voice: "My adored one, I love you wildly. Since yesterday I have been suffering the tortures of the damned, haunted by our memory. I feel your lips against mine, your eyes in mine, your breast against mine. I love you, I love you! You have driven me mad. My arms open, I gasp, moved by a wild desire to hold you again. My whole soul and body cries out for you, wants you. I have kept in my mouth the taste of your kisses--"
The judge had straightened himself up. The nun stopped reading. He snatched the letter from her and looked for the signature. There was none, but only under the words, "The man who adores you," the name "Henry." Their father's name was Rene. Therefore this was not from him. The son then quickly rummaged through the package of letters, took one out and read: "I can no longer live without your caresses." Standing erect, severe as when sitting on the bench, he looked unmoved at the dead woman. The nun, straight as a statue, tears trembling in the corners of her eyes, was watching her brother, waiting. Then he crossed the room slowly, went to the window and stood there, gazing out into the dark night.
When he turned around again Sister Eulalie, her eyes dry now, was still standing near the bed, her head bent down.
He stepped forward, quickly picked up the letters and threw them pell-mell back into the drawer. Then he closed the curtains of the bed.
When daylight made the candles on the table turn pale the son slowly left his armchair, and without looking again at the mother upon whom he had passed sentence, severing the tie that united her to son and daughter, he said slowly: "Let us now retire, sister."
"""
story_2 = """Every year Thanksgiving night we flocked out behind Dad as he dragged the Santa suit to the road and draped it over a kind of crucifix he'd built out of metal pole in the yard. Super Bowl week the pole was dressed in a jersey and Rod's helmet and Rod had to clear it with Dad if he wanted to take the helmet off. On the Fourth of July the pole was Uncle Sam, on Veteran’s Day a soldier,  on Halloween a ghost. The pole was Dad's only concession to glee. We were allowed a single Crayola from the box at a time. One Christmas Eve he shrieked at Kimmie for wasting an apple slice. He hovered over us as we poured ketchup saying: good enough good enough good enough. Birthday parties consisted of cupcakes, no ice cream. The first time I brought a date over she said: what's with your dad and that pole? and I sat there blinking.

We left home, married,  had children of our own, found the seeds of meanness blooming also within us. Dad began dressing the pole with more complexity and less discernible logic. He draped some kind of fur over it on Groundhog Day and lugged out a floodlight to ensure a shadow. When an earthquake struck Chile he lay the pole on its side and spray painted a rift in the earth. Mom died and he dressed the pole as Death and hung from the crossbar photos of Mom as a baby. We'd stop by and find odd talismans from his youth arranged around the base: army medals, theater tickets, old sweatshirts, tubes of Mom's makeup. One autumn he painted the pole bright yellow. He covered it with cotton swabs that winter for warmth and provided offspring by hammering in six crossed sticks around the yard. He ran lengths of string between the pole and the sticks, and taped to the string letters of apology, admissions of error, pleas for understanding, all written in a frantic hand on index cards. He painted a sign saying LOVE and hung it from the pole and another that said FORGIVE? and then he died in the hall with the radio on and we sold the house to a young couple who yanked out the pole and the sticks and left them by the road on garbage day."""

story_3 = """A guard came to the prison shoe-shop, where Jimmy Valentine was assiduously stitching uppers, and escorted him to the front office. There the warden handed Jimmy his pardon, which had been signed that morning by the governor. Jimmy took it in a tired kind of way. He had served nearly ten months of a four year sentence. He had expected to stay only about three months, at the longest. When a man with as many friends on the outside as Jimmy Valentine had is received in the "stir" it is hardly worth while to cut his hair.

"Now, Valentine," said the warden, "you'll go out in the morning. Brace up, and make a man of yourself. You're not a bad fellow at heart. Stop cracking safes, and live straight."

"Me?" said Jimmy, in surprise. "Why, I never cracked a safe in my life."

"Oh, no," laughed the warden. "Of course not. Let's see, now. How was it you happened to get sent up on that Springfield job? Was it because you wouldn't prove an alibi for fear of compromising somebody in extremely high-toned society? Or was it simply a case of a mean old jury that had it in for you? It's always one or the other with you innocent victims."

"Me?" said Jimmy, still blankly virtuous. "Why, warden, I never was in Springfield in my life!"

"Take him back, Cronin!" said the warden, "and fix him up with outgoing clothes. Unlock him at seven in the morning, and let him come to the bull-pen. Better think over my advice, Valentine."

At a quarter past seven on the next morning Jimmy stood in the warden's outer office. He had on a suit of the villainously fitting, ready-made clothes and a pair of the stiff, squeaky shoes that the state furnishes to its discharged compulsory guests.

The clerk handed him a railroad ticket and the five-dollar bill with which the law expected him to rehabilitate himself into good citizenship and prosperity. The warden gave him a cigar, and shook hands. Valentine, 9762, was chronicled on the books, "Pardoned by Governor," and Mr. James Valentine walked out into the sunshine.

Disregarding the song of the birds, the waving green trees, and the smell of the flowers, Jimmy headed straight for a restaurant. There he tasted the first sweet joys of liberty in the shape of a broiled chicken and a bottle of white wine—followed by a cigar a grade better than the one the warden had given him. From there he proceeded leisurely to the depot. He tossed a quarter into the hat of a blind man sitting by the door, and boarded his train. Three hours set him down in a little town near the state line. He went to the café of one Mike Dolan and shook hands with Mike, who was alone behind the bar.

"Sorry we couldn't make it sooner, Jimmy, me boy," said Mike. "But we had that protest from Springfield to buck against, and the governor nearly balked. Feeling all right?"

"Fine," said Jimmy. "Got my key?"

He got his key and went upstairs, unlocking the door of a room at the rear. Everything was just as he had left it. There on the floor was still Ben Price's collar-button that had been torn from that eminent detective's shirt-band when they had overpowered Jimmy to arrest him.

Pulling out from the wall a folding-bed, Jimmy slid back a panel in the wall and dragged out a dust-covered suit-case. He opened this and gazed fondly at the finest set of burglar's tools in the East. It was a complete set, made of specially tempered steel, the latest designs in drills, punches, braces and bits, jimmies, clamps, and augers, with two or three novelties, invented by Jimmy himself, in which he took pride. Over nine hundred dollars they had cost him to have made at ----, a place where they make such things for the profession.

In half an hour Jimmy went down stairs and through the café. He was now dressed in tasteful and well-fitting clothes, and carried his dusted and cleaned suit-case in his hand.

"Got anything on?" asked Mike Dolan, genially.

"Me?" said Jimmy, in a puzzled tone. "I don't understand. I'm representing the New York Amalgamated Short Snap Biscuit Cracker and Frazzled Wheat Company."

This statement delighted Mike to such an extent that Jimmy had to take a seltzer-and-milk on the spot. He never touched "hard" drinks.

A week after the release of Valentine, 9762, there was a neat job of safe-burglary done in Richmond, Indiana, with no clue to the author. A scant eight hundred dollars was all that was secured. Two weeks after that a patented, improved, burglar-proof safe in Logansport was opened like a cheese to the tune of fifteen hundred dollars, currency; securities and silver untouched. That began to interest the rogue-catchers. Then an old-fashioned bank-safe in Jefferson City became active and threw out of its crater an eruption of bank-notes amounting to five thousand dollars. The losses were now high enough to bring the matter up into Ben Price's class of work. By comparing notes, a remarkable similarity in the methods of the burglaries was noticed. Ben Price investigated the scenes of the robberies, and was heard to remark:

"That's Dandy Jim Valentine's autograph. He's resumed business. Look at that combination knob—jerked out as easy as pulling up a radish in wet weather. He's got the only clamps that can do it. And look how clean those tumblers were punched out! Jimmy never has to drill but one hole. Yes, I guess I want Mr. Valentine. He'll do his bit next time without any short-time or clemency foolishness."

Ben Price knew Jimmy's habits. He had learned them while working up the Springfield case. Long jumps, quick get-aways, no confederates, and a taste for good society—these ways had helped Mr. Valentine to become noted as a successful dodger of retribution. It was given out that Ben Price had taken up the trail of the elusive cracksman, and other people with burglar-proof safes felt more at ease.

One afternoon Jimmy Valentine and his suit-case climbed out of the mail-hack in Elmore, a little town five miles off the railroad down in the black-jack country of Arkansas. Jimmy, looking like an athletic young senior just home from college, went down the board side-walk toward the hotel.

A young lady crossed the street, passed him at the corner and entered a door over which was the sign, "The Elmore Bank." Jimmy Valentine looked into her eyes, forgot what he was, and became another man. She lowered her eyes and coloured slightly. Young men of Jimmy's style and looks were scarce in Elmore.

Jimmy collared a boy that was loafing on the steps of the bank as if he were one of the stockholders, and began to ask him questions about the town, feeding him dimes at intervals. By and by the young lady came out, looking royally unconscious of the young man with the suit-case, and went her way.

"Isn't that young lady Polly Simpson?" asked Jimmy, with specious guile.

"Naw," said the boy. "She's Annabel Adams. Her pa owns this bank. What'd you come to Elmore for? Is that a gold watch-chain? I'm going to get a bulldog. Got any more dimes?"

Jimmy went to the Planters' Hotel, registered as Ralph D. Spencer, and engaged a room. He leaned on the desk and declared his platform to the clerk. He said he had come to Elmore to look for a location to go into business. How was the shoe business, now, in the town? He had thought of the shoe business. Was there an opening?

The clerk was impressed by the clothes and manner of Jimmy. He, himself, was something of a pattern of fashion to the thinly gilded youth of Elmore, but he now perceived his shortcomings. While trying to figure out Jimmy's manner of tying his four-in-hand he cordially gave information.

Yes, there ought to be a good opening in the shoe line. There wasn't an exclusive shoe-store in the place. The dry-goods and general stores handled them. Business in all lines was fairly good. Hoped Mr. Spencer would decide to locate in Elmore. He would find it a pleasant town to live in, and the people very sociable.

Mr. Spencer thought he would stop over in the town a few days and look over the situation. No, the clerk needn't call the boy. He would carry up his suit-case, himself; it was rather heavy.

Mr. Ralph Spencer, the phœnix that arose from Jimmy Valentine's ashes—ashes left by the flame of a sudden and alterative attack of love—remained in Elmore, and prospered. He opened a shoe-store and secured a good run of trade.

Socially he was also a success, and made many friends. And he accomplished the wish of his heart. He met Miss Annabel Adams, and became more and more captivated by her charms.

At the end of a year the situation of Mr. Ralph Spencer was this: he had won the respect of the community, his shoe-store was flourishing, and he and Annabel were engaged to be married in two weeks. Mr. Adams, the typical, plodding, country banker, approved of Spencer. Annabel's pride in him almost equalled her affection. He was as much at home in the family of Mr. Adams and that of Annabel's married sister as if he were already a member.

One day Jimmy sat down in his room and wrote this letter, which he mailed to the safe address of one of his old friends in St. Louis:
 

        Dear Old Pal:

        I want you to be at Sullivan's place, in Little Rock, next Wednesday night, at nine o'clock. I want you to wind up some little matters for me. And, also, I want to make you a present of my kit of tools. I know you'll be glad to get them—you couldn't duplicate the lot for a thousand dollars. Say, Billy, I've quit the old business—a year ago. I've got a nice store. I'm making an honest living, and I'm going to marry the finest girl on earth two weeks from now. It's the only life, Billy—the straight one. I wouldn't touch a dollar of another man's money now for a million. After I get married I'm going to sell out and go West, where there won't be so much danger of having old scores brought up against me. I tell you, Billy, she's an angel. She believes in me; and I wouldn't do another crooked thing for the whole world. Be sure to be at Sully's, for I must see you. I'll bring along the tools with me.

        Your old friend,

        Jimmy.
         

On the Monday night after Jimmy wrote this letter, Ben Price jogged unobtrusively into Elmore in a livery buggy. He lounged about town in his quiet way until he found out what he wanted to know. From the drug-store across the street from Spencer's shoe-store he got a good look at Ralph D. Spencer.

"Going to marry the banker's daughter are you, Jimmy?" said Ben to himself, softly. "Well, I don't know!"

The next morning Jimmy took breakfast at the Adamses. He was going to Little Rock that day to order his wedding-suit and buy something nice for Annabel. That would be the first time he had left town since he came to Elmore. It had been more than a year now since those last professional "jobs," and he thought he could safely venture out.

After breakfast quite a family party went downtown together—Mr. Adams, Annabel, Jimmy, and Annabel's married sister with her two little girls, aged five and nine. They came by the hotel where Jimmy still boarded, and he ran up to his room and brought along his suit-case. Then they went on to the bank. There stood Jimmy's horse and buggy and Dolph Gibson, who was going to drive him over to the railroad station.

All went inside the high, carved oak railings into the banking-room—Jimmy included, for Mr. Adams's future son-in-law was welcome anywhere. The clerks were pleased to be greeted by the good-looking, agreeable young man who was going to marry Miss Annabel. Jimmy set his suit-case down. Annabel, whose heart was bubbling with happiness and lively youth, put on Jimmy's hat, and picked up the suit-case. "Wouldn't I make a nice drummer?" said Annabel. "My! Ralph, how heavy it is? Feels like it was full of gold bricks."

"Lot of nickel-plated shoe-horns in there," said Jimmy, coolly, "that I'm going to return. Thought I'd save express charges by taking them up. I'm getting awfully economical."

The Elmore Bank had just put in a new safe and vault. Mr. Adams was very proud of it, and insisted on an inspection by every one. The vault was a small one, but it had a new, patented door. It fastened with three solid steel bolts thrown simultaneously with a single handle, and had a time-lock. Mr. Adams beamingly explained its workings to Mr. Spencer, who showed a courteous but not too intelligent interest. The two children, May and Agatha, were delighted by the shining metal and funny clock and knobs.

While they were thus engaged Ben Price sauntered in and leaned on his elbow, looking casually inside between the railings. He told the teller that he didn't want anything; he was just waiting for a man he knew.

Suddenly there was a scream or two from the women, and a commotion. Unperceived by the elders, May, the nine-year-old girl, in a spirit of play, had shut Agatha in the vault. She had then shot the bolts and turned the knob of the combination as she had seen Mr. Adams do.

The old banker sprang to the handle and tugged at it for a moment. "The door can't be opened," he groaned. "The clock hasn't been wound nor the combination set."

Agatha's mother screamed again, hysterically.

"Hush!" said Mr. Adams, raising his trembling hand. "All be quite for a moment. Agatha!" he called as loudly as he could. "Listen to me." During the following silence they could just hear the faint sound of the child wildly shrieking in the dark vault in a panic of terror.

"My precious darling!" wailed the mother. "She will die of fright! Open the door! Oh, break it open! Can't you men do something?"

"There isn't a man nearer than Little Rock who can open that door," said Mr. Adams, in a shaky voice. "My God! Spencer, what shall we do? That child—she can't stand it long in there. There isn't enough air, and, besides, she'll go into convulsions from fright."

Agatha's mother, frantic now, beat the door of the vault with her hands. Somebody wildly suggested dynamite. Annabel turned to Jimmy, her large eyes full of anguish, but not yet despairing. To a woman nothing seems quite impossible to the powers of the man she worships.

"Can't you do something, Ralph — try, won't you?"

He looked at her with a queer, soft smile on his lips and in his keen eyes.

"Annabel," he said, "give me that rose you are wearing, will you?"

Hardly believing that she heard him aright, she unpinned the bud from the bosom of her dress, and placed it in his hand. Jimmy stuffed it into his vest-pocket, threw off his coat and pulled up his shirt-sleeves. With that act Ralph D. Spencer passed away and Jimmy Valentine took his place.

"Get away from the door, all of you," he commanded, shortly.

He set his suit-case on the table, and opened it out flat. From that time on he seemed to be unconscious of the presence of any one else. He laid out the shining, queer implements swiftly and orderly, whistling softly to himself as he always did when at work. In a deep silence and immovable, the others watched him as if under a spell.

In a minute Jimmy's pet drill was biting smoothly into the steel door. In ten minutes—breaking his own burglarious record—he threw back the bolts and opened the door.

Agatha, almost collapsed, but safe, was gathered into her mother's arms.

Jimmy Valentine put on his coat, and walked outside the railings towards the front door. As he went he thought he heard a far-away voice that he once knew call "Ralph!" But he never hesitated.

At the door a big man stood somewhat in his way.

"Hello, Ben!" said Jimmy, still with his strange smile. "Got around at last, have you? Well, let's go. I don't know that it makes much difference, now."

And then Ben Price acted rather strangely.

"Guess you're mistaken, Mr. Spencer," he said. "Don't believe I recognize you. Your buggy's waiting for you, ain't it?"

And Ben Price turned and strolled down the street.
"""

story_4 = """The gunman lights a cigarette, watches despondently as dusk falls upon the empty alley. He is alone in a lonely place, summoned here to receive instructions from a master criminal known only as the Boss, but the Boss isn’t here. No one is. It’s spooky. He feels like a marked man. The Boss is known for his ruthlessness. When he orders a killing, someone dies. The gunman would like there to be witnesses for what happens next, but the alley’s deserted.

He glances at his watch, a gift from the Boss. Face a gold coin, no numbers. A joke, probably: time is money. Or, maybe, money is time; it depends on what you’re short of. The Boss is a great joker. The watch hands are hair-thin, like the edge of a razor blade, hard to see, especially in this fading light. There and not there, like time itself. Which is perhaps not being clocked—perhaps that’s what the numberless face is saying. How can you measure the shit you’re buried in? He doesn’t know what keeps the watch running. Battery inside, maybe. When the battery dies? Don’t think about it.

On one assignment or another, the gunman has wasted a few suckers in alleys like this, but not many. His specialty is knocking them off in train stations and hotel lobbies or on the street in broad daylight. There’s a greater risk of getting nabbed, so always an extra thrill. Which is why he took up doing the Boss’s dirty work. Thrills remind him that he’s alive, when not much else does.

Overhead, a street lamp comes on, its yellow light a mere stain under the dimming sky. Then a woman passes. It’s about time, the gunman thinks. Has she been sent by the Boss? Is she his witness? His victim? His executioner? His hand is inside his jacket, resting on the butt of the revolver holstered under his armpit. Probably just an innocent working girl, looking for a pickup, wandering into a place where she doesn’t belong. Which wouldn’t stop the Boss from targeting her. Or using her.

You looking for someone, sister? he asks, lurking in the shadows, his cigarette bobbing as he speaks.

I don’t know, she says. She casts an inquisitive gaze at him, then quickly looks away. A signal? He should perhaps throw himself behind the trash cans, but her tender scrutiny has fixed him where he stands.

She walks slowly away into the shadows at the far end of the alley, then swivels and walks back again toward the light. She appears frightened, lost. Touchingly vulnerable. Above her, the street lamp sways rhythmically, causing shadows to stretch and recede, stretch and recede, like a slowly beating heart. She pauses under it and looks around, clearly thinking about something. She’s no beauty, but she has sweet ways. His hard heart softens. He takes his hand off the revolver.

When she turns to walk down the alley again, the gunman flicks his cigarette away and walks with her, moving as she moves. Left, right, left . . . it’s a kind of dance. She asks if he’s making fun of her. He says he doesn’t know how to make fun. In truth, he feels caught up in something fundamental. Benumbing. A feeling he hasn’t had before. Abstract, yet oddly erotic.

It won’t do any good, she says, reading him.

Yeah, I know, lady. But I like it.

They reach the dark end of the alley, still moving together, and turn back toward the swaying street lamp. The alley seems almost to fade away, the lamp’s glow to brighten. There are bats twittering somewhere. Time itself seems to stop, even though he knows it’s still grinding inexorably toward something messy. The Boss has plans that haven’t happened yet. Never mind. Worry about them when they do.

When their dance ends, night has fully descended, sharpening the contrast between the pool of lamplight they have stepped into and the rest of the world, lost in the darkness beyond. That was great, he says. She looks up at him, her face pale but radiant in the light. She nods. Her sad smile seems to say, Yes, but it’s not enough, is it? She stares down at her feet, and when she does so, her face is masked in shadow. Is she about to betray him? he wonders. Does he care?

She starts suddenly, glances back down the alley. He follows her gaze. Uh-oh. Something is moving. His hand, too slowly, is inside his jacket. Watch out! she cries, and throws herself in front of him. A shot rings out. She collapses into his arms. Holding her, he fires impotently into the darkness. His bullets ricochet through the alley like ironic laughter. Hang on, baby! he pleads, but it’s too late.

It must be . . . why . . . she whispers, and dies in his arms.

His heart hardens again. Rage bites at the corners of his eyes. He doesn’t understand a fucking thing."""

story_5 = """The husband had got into the habit of calling the wife from somewhere in the house—if she was upstairs, he was downstairs; if she was downstairs, he was upstairs—and when she answered, “Yes? What?,” he would continue to call her, as if he hadn’t heard and with an air of strained patience: “Hello? Hello? Where are you?” And so she had no choice but to hurry to him, wherever he was, elsewhere in the house, downstairs, upstairs, in the basement or outside on the deck, in the back yard or in the driveway. “Yes?” she called, trying to remain calm. “What is it?” And he would tell her—a complaint, a remark, an observation, a reminder, a query—and then, later, she would hear him calling again with a new urgency, “Hello? Hello? Where are you?,” and she would call back, “Yes? What is it?,” trying to determine where he was. He would continue to call, not hearing her, for he disliked wearing his hearing aid around the house, where there was only the wife to be heard. He complained that one of the little plastic devices in the shape of a snail hurt his ear, the tender inner ear was reddened and had even bled, and so he would call, pettishly, “Hello? Where are you?”—for the woman was always going off somewhere out of the range of his hearing, and he never knew where the hell she was or what she was doing; at times, her very being exasperated him—until finally she gave in and ran breathless to search for him, and when he saw her he said reproachfully, “Where were you? I worry about you when you don’t answer.” And she said, laughing, trying to laugh, though none of this was funny, “But I was here all along!” And he retorted, “No, you were not. You were not. I was here, and you were not here.” And later that day, after his lunch and before his nap, unless it was before his lunch and after his nap, the wife heard the husband calling to her, “Hello? Hello? Where are you?,” and the thought came to her, No. I will hide from him. But she would not do such a childish thing. Instead she stood on the stairs and cupped her hands to her mouth and called to him, “I’m here. I’m always here. Where else would I be?” But the husband couldn’t hear her and continued to call, “Hello? Hello? Where are you?,” until at last she screamed, “What do you want? I’ve told you, I’m here.” But the husband couldn’t hear and continued to call, “Hello? Where are you? Hello!,” and finally the wife had no choice but to give in, for the husband was sounding vexed and angry and anxious. Descending the stairs, she tripped and fell, fell hard, and her neck was broken in an instant, and she died at the foot of the stairs, while in one of the downstairs rooms, or perhaps in the cellar, or on the deck at the rear of the house, the husband continued to call, with mounting urgency, “Hello? Hello? Where are you?”"""

story_6 = """It doesn’t make the sound that you think it would make. I mean, I figured it would be loud, or top-heavy. But it sounded like almost nothing, like water dripping from a shower faucet three rooms away. The only reason I knew what had happened at all was the communion table splitting clean in half, scattering the grape juice and the crackers everywhere.

I stepped over Jesus’ body and his blood on my way out.

The church was the biggest building in our whole neighborhood, which is to say, it was small, so small the mural painter could only fit two members of the Holy Trinity onto the biggest wall. My father had to reassure the town that the Holy Ghost was invisible, anyway, and lived inside of all of us, so, it was fine either way. My father was the head pastor. And on Sunday morning the town—all three hundred and two members—entered church and saw the broken communion table and all that spilt body and blood.

Well, there were murmurs.

Martha Lou, rubbing coconut lotion into her palms: “I mean, I’m not one for superstition but, we find little Mary’s body, and then all this madness with the table—!”

Bobby Greene, with an only half secured toupee: “—I heard Mary was with child, that tiny little thing, and what, only nineteen—?!”

Jilly Jean, who carried washed Tupperware in her purse, just in case: “That little thing? Not even old enough to be married, much less—well, and just the other day she was baptized by the head pastor himself, practically reciting his sermons verbatim, and then all this—”

Bill Lewis, who routinely lied about golfing on the weekends: “Pregnant and everything, oh, the poor thing, well, I’m sure this isn’t the first young unmarried girl who winds up with a baby and then, you know, winds up dead, as horrible as it sounds, I mean, you know—”

And they did know. Everyone knew.

It got so bad, during his sermon, my father threw in a line about faith, and always having it, even when strange things like young girls dying too soon and communion tables breaking clean in half happened. And that was that: my father had spoken. The murmurs stilled almost instantly.

Father, gruffly: “Son, hand me that piece of wood.”

He was making a new communion table. Corkscrew wood shavings littered the floor—tiny pieces of holy to be swept up and thrown away. He was building God from scratch—tiny pieces of conviction and power that would be collected with the care of a creator molding his creation, to be put into the pockets of people’s minds until they forgot that it was there altogether.

From outside the church window, you could see the track field. Four boys, lanky and tan, were racing against each other. First one pulled ahead, then another, then another. Girls on the sidelines stood cheering, occasionally ripping up grass and throwing it at each other. Finally, one of the boys—the one with long hair—won. He sprawled out on the track, chest heaving. I watched him throw his arms up in the air—another triumph over the invisible ghosts in our sky.

Me, the son: “You’ve heard the rumors, right?”

Father: “The Bible wasn’t written on rumors.”

Me, ignoring Father: “They’re saying that Mary was pregnant when she died.”

Father continued sanding.

Me: “And I’m thinking, well, I’m thinking about all the times we had her over. Especially when mom was away. I’m thinking, like, she stayed late a lot, and, you were gone a lot, with her, I’m guessing, and well, now she’s dead. And so, I guess—I guess I’m thinking, it was you and it was your baby, and well—well, I’ve heard the stories, you know, about unmarried girls and their babies and the people with reputations to keep who kill them. So.”

Father had stopped sanding the table. His hands were shaking. Hard. He said nothing.

Then, Me, again: “I went into church yesterday, after they found it, you know, her body? And I think I heard—I think I heard god. Except, he didn’t say anything, it just sounded like he was running. Running away. From you—from us—from everything.”

There was a silence. Well, an almost silence. I could hear something like a shower faucet, dripping three—maybe even six or seven—rooms away."""

story_7 = """Monday dawned warm and rainless. Aurelio Escovar, a dentist without a degree, and a very early riser, opened his office at six. He took some false teeth, still mounted in their plaster mold, out of the glass case and put on the table a fistful of instruments which he arranged in size order, as if they were on display. He wore a collarless striped shirt, closed at the neck with a golden stud, and pants held up by suspenders He was erect and skinny, with a look that rarely corresponded to the situation, the way deaf people have of looking.
When he had things arranged on the table, he pulled the drill toward the dental chair and sat down to polish the false teeth. He seemed not to be thinking about what he was doing, but worked steadily, pumping the drill with his feet, even when he didn't need it.
After eight he stopped for a while to look at the sky through the window, and he saw two pensive buzzards who were drying themselves in the sun on the ridgepole of the house next door. He went on working with the idea that before lunch it would rain again. The shrill voice of his elevenyear-old son interrupted his concentration.
"Papa."
"What?"
"The Mayor wants to know if you'll pull his tooth."
"Tell him I'm not here."
He was polishing a gold tooth. He held it at arm's length, and examined it with his eyes half closed. His son shouted again from the little waiting room.
"He says you are, too, because he can hear you."
The dentist kept examining the tooth. Only when he had put it on the table with the finished work did he say:
"So much the better."
He operated the drill again. He took several pieces of a bridge out of a cardboard box where he kept the things he still had to do and began to polish the gold.
"Papa."
"What?"
He still hadn't changed his expression.
"He says if you don't take out his tooth, he'll shoot you."
Without hurrying, with an extremely tranquil movement, he stopped pedaling the drill, pushed it away from the chair, and pulled the lower drawer of the table all the way out. There was a revolver. "O.K.," he said. "Tell him to come and shoot me."
He rolled the chair over opposite the door, his hand resting on the edge of the drawer. The Mayor appeared at the door. He had shaved the left side of his face, but the other side, swollen and in pain, had a five-day-old beard. The dentist saw many nights of desperation in his dull eyes. He closed the drawer with his fingertips and said softly:
"Sit down."
"Good morning," said the Mayor.
"Morning," said the dentist.
While the instruments were boiling, the Mayor leaned his skull on the headrest of the chair and felt better. His breath was icy. It was a poor office: an old wooden chair, the pedal drill, a glass case with ceramic bottles. Opposite the chair was a window with a shoulder-high cloth curtain. When he felt the dentist approach, the Mayor braced his heels and opened his mouth.
Aurelio Escovar turned his head toward the light. After inspecting the infected tooth, he closed the Mayor's jaw with a cautious pressure of his fingers.
"It has to be without anesthesia," he said.
"Why?"
"Because you have an abscess."
The Mayor looked him in the eye. "All right," he said, and tried to smile. The dentist did not return the smile. He brought the basin of sterilized instruments to the worktable and took them out of the water with a pair of cold tweezers, still without hurrying. Then he pushed the spittoon with the tip of his shoe, and went to wash his hands in the washbasin. He did all this without looking at the Mayor. But the Mayor didn't take his eyes off him.
It was a lower wisdom tooth. The dentist spread his feet and grasped the tooth with the hot forceps. The Mayor seized the arms of the chair, braced his feet with all his strength, and felt an icy void in his kidneys, but didn't make a sound. The dentist moved only his wrist. Without rancor, rather with a bitter tenderness, he said:
"Now you'll pay for our twenty dead men."
The Mayor felt the crunch of bones in his jaw, and his eyes filled with tears. But he didn't breathe until he felt the tooth come out. Then he saw it through his tears. It seemed so foreign to his pain that he failed to understand his torture of the five previous nights.
Bent over the spittoon, sweating, panting, he unbuttoned his tunic and reached for the handkerchief in his pants pocket. The dentist gave him a clean cloth.
"Dry your tears," he said.
The Mayor did. He was trembling. While the dentist washed his hands, he saw the crumbling ceiling and a dusty spider web with spider's eggs and dead insects. The dentist returned, drying his hands. "Go to bed," he said, "and gargle with salt water." The Mayor stood up, said goodbye with a casual military salute, and walked toward the door, stretching his legs, without buttoning up his tunic.
"Send the bill," he said.
"To you or the town?"
The Mayor didn't look at him. He closed the door and said through the screen:
"It's the same damn thing."
"""

story_8 = """Whatever hour you woke there was a door shutting. From room to room they went, hand in hand, lifting here, opening there, making sure--a ghostly couple.
"Here we left it," she said. And he added, "Oh, but here tool" "It's upstairs," she murmured. "And in the garden," he whispered. "Quietly," they said, "or we shall wake them."
But it wasn't that you woke us. Oh, no. "They're looking for it; they're drawing the curtain," one might say, and so read on a page or two. "Now they've found it,' one would be certain, stopping the pencil on the margin. And then, tired of reading, one might rise and see for oneself, the house all empty, the doors standing open, only the wood pigeons bubbling with content and the hum of the threshing machine sounding from the farm. "What did I come in here for? What did I want to find?" My hands were empty. "Perhaps its upstairs then?" The apples were in the loft. And so down again, the garden still as ever, only the book had slipped into the grass.
But they had found it in the drawing room. Not that one could ever see them. The windowpanes reflected apples, reflected roses; all the leaves were green in the glass. If they moved in the drawing room, the apple only turned its yellow side. Yet, the moment after, if the door was opened, spread about the floor, hung upon the walls, pendant from the ceiling--what? My hands were empty. The shadow of a thrush crossed the carpet; from the deepest wells of silence the wood pigeon drew its bubble of sound. "Safe, safe, safe" the pulse of the house beat softly. "The treasure buried; the room . . ." the pulse stopped short. Oh, was that the buried treasure?
A moment later the light had faded. Out in the garden then? But the trees spun darkness for a wandering beam of sun. So fine, so rare, coolly sunk beneath the surface the beam I sought always burned behind the glass. Death was the glass; death was between us, coming to the woman first, hundreds of years ago, leaving the house, sealing all the windows; the rooms were darkened. He left it, left her, went North, went East, saw the stars turned in the Southern sky; sought the house, found it dropped beneath the Downs. "Safe, safe, safe," the pulse of the house beat gladly. 'The Treasure yours."
The wind roars up the avenue. Trees stoop and bend this way and that. Moonbeams splash and spill wildly in the rain. But the beam of the lamp falls straight from the window. The candle burns stiff and still. Wandering through the house, opening the windows, whispering not to wake us, the ghostly couple seek their joy.
"Here we slept," she says. And he adds, "Kisses without number." "Waking in the morning--" "Silver between the trees--" "Upstairs--" 'In the garden--" "When summer came--" 'In winter snowtime--" "The doors go shutting far in the distance, gently knocking like the pulse of a heart.
Nearer they come, cease at the doorway. The wind falls, the rain slides silver down the glass. Our eyes darken, we hear no steps beside us; we see no lady spread her ghostly cloak. His hands shield the lantern. "Look," he breathes. "Sound asleep. Love upon their lips."
Stooping, holding their silver lamp above us, long they look and deeply. Long they pause. The wind drives straightly; the flame stoops slightly. Wild beams of moonlight cross both floor and wall, and, meeting, stain the faces bent; the faces pondering; the faces that search the sleepers and seek their hidden joy.
"Safe, safe, safe," the heart of the house beats proudly. "Long years--" he sighs. "Again you found me." "Here," she murmurs, "sleeping; in the garden reading; laughing, rolling apples in the loft. Here we left our treasure--" Stooping, their light lifts the lids upon my eyes. "Safe! safe! safe!" the pulse of the house beats wildly. Waking, I cry "Oh, is this your buried treasure? The light in the heart."
"""

story_9 = """Near our mountain cabin, in Jahorina, there was once a hotel called Šator. It was open only in the winter for the skiing season. When you stood outside the hotel under a frigid, starry sky, you could smell cafeteria grease, wood fire, and cigarette smoke, and hear the thumping from the disco club in the basement. For the rest of the year, the hotel was vacant. One summer, when I was eleven, I broke into the hotel bar by cracking open its window—it took me an hour—and stole a bottle of blueberry juice. A guy who was a bartender there in the winter, but idled in the summer, caught me and blackmailed me, asking for money not to tell my parents. I told him to fuck off. He told my parents. They punished me, but that blueberry juice was the sweetest of potions. The hotel’s cleaning staff was a woman of undetermined body shape and age, named Baja, who spent summer days sitting on the hotel balcony, looking at nothing, always wearing a blue overcoat and a black scarf on her head, one of its corners covering her jaw to keep it warm. She was, like a ghost, impervious to pain. She had an eternal toothache that she would not treat, the abscess swelling until it had devoured and destroyed one of her eyes. Nobody ever saw her clean anything, though a friend of mine who’d stayed at the hotel told me that Baja had once walked into his room without knocking, looked around, and said, “Why don’t you clean this up? It’s disgusting.” Right behind the hotel, there was a crested boulder, which my sister and I would climb when there was nothing else to do. From the top, we could see the weekend-house cluster below and hear the buzzing of circular saws, the banging of hammers, the din of aspiration, for the weekenders were perpetually getting ready for some future in which active life would be successfully completed and there’d be nothing but peace, virginal nature, and retirement. The war would put an end to that ambition, but back then, when we were kids, the future was always on its way, its advance currents flowing through everything we knew or wanted to know. If we looked away from the house cluster, wooded vales and peaks, meadows and roads stretched toward Sarajevo and beyond, onward to the horizon, into which, at the end of the day, the sun would slide like a coin into a slot. We’d stand at the edge of the boulder, pine tips emerging from the verdant void beneath our feet, and we’d look, and look, and look: our visual field had no limits, just as our life had no end. The garbage from the hotel was dumped right behind it, down the slope at the foot of the boulder. My sister and I didn’t find that strange. By the time we emerged from the unconscious part of our childhood, the world seemed fully established, everything as it was supposed to be, all the points and objects fixed, all the hierarchies and structures natural and unalterable. We’d descend from the peak to the garbage dump to browse through trash-stuffed bags and rotten food remnants. One day, we found a plethora of plates, saucers, cups, and bowls strewn all over the dump—the hotel had replaced its dishes and discarded the old ones. We spent the whole day breaking those dishes with rocks or against one another. We discussed nothing, devised no plans; it was clear to us what needed to be done. The dishes were nondescript, beige. We shattered them into smithereens, taking a break for lunch, then smashing some more in the afternoon. We sustained small cuts, but we didn’t care. They were the blisters of toil, the stigmata of devotion. We discovered the pleasure of unbridled, unlimited destruction, the endless joy of converting something into nothing. This was new to us back then. Now it isn’t. Now we know that that was one of the happiest days of our childhood, perhaps of our entire life. And when we went back to the dump, some time later, and found new garbage there, new generations of refuse, we knew that underneath it all were our smithereens, that we could go on forging them for as long as we were alive, that we would always remember the day we first broke the limited whole."""

system_prompt_1 = """You will be given a story. You always output a list of "all" the tropes you found in the context of the story."""

system_prompt_2 = "You will be given a story and you must list all the tropes you found in the story."

system_prompt_3 = "You are an expert on the TV Tropes dataset. You will be given a story. You always output all the tropes you found in the story."

system_prompt_4 = """You are an expert on tropes. You will be given a story. You output what is asked of you in the user message."""

system_prompt_5 = "You are an expert on tropes. You will be given a story in INST block. Your task is to extract all the tropes in the story."

system_prompt_6 = "Analyze the story given in <</SYS>> and [/INST]. List ALL the tropes you found after the analysis and also output the reason why you think the trope exists in the story. Perform a deep dive analysis and not limit your output to 10 tropes."

system_prompt_7 = "You will be given the text of a short story within <</SYS>> and [/INST]. Perform a deep dive analysis of the story and list ALL the tropes you can find in the story."

system_prompt_8 = """You will be given a story. You perform a deep analysis of the story and always output a list of "all" the tropes you found in the context of the story."""

system_prompt_9 = "You are an expert at finding tropes in narratives. You will be given a story. You perform a deep analysis of the story and list all the tropes you found in the story."

system_prompt_10 = "You are an expert on the TV Tropes dataset. You will be given a story. You perform a deep analysis of the story and output all the tropes you found in the story."

user_message_1 = "{}"

user_message_2 = "List all the tropes you can find in the story below -\n {}"

user_message_3 = """List all the tropes you can find after analyzing the story that is in between the delimiters ---.\n
---
{}
---"""

system_prompts = [system_prompt_1, system_prompt_2, system_prompt_3, system_prompt_4, system_prompt_5, system_prompt_6, system_prompt_7, system_prompt_8, system_prompt_9, system_prompt_10]
user_messages = [user_message_1, user_message_2, user_message_3]
stories = [story_1, story_2, story_3, story_4, story_5, story_6, story_7, story_8, story_9]

prompt = """<s>[INST] <<SYS>>
{}
<</SYS>>

{} [/INST]"""

# for temp in temperatures:
    # pipe = pipe(temperature=temp)
print(model.generation_config)
print("######")

for story in stories:
    for system_p in system_prompts:
        for user_m in user_messages:
            input = prompt.format(system_p.strip(), user_m.format(story.strip()))
        # print(system_p)
        # print(user_m)
            # generate(input, temp)
            generate(input)
            # pipe_generate(input, temperature=temp)
            # pipe_generate(input)
            print("###################################")

