from utils_evaluate import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # path of the test file
    filepath = 'data/test_medium.txt'

    # creation and transformation of the input test file
    test_df = create_df(filepath)
    test_df = adapt_df(test_df)

    # drop the index column (If we want a fraction of the test just modify frac parameter in sample)
    test_df = test_df.reset_index(drop=True)


    MODEL_NAME = "gpt2"
    MAX_LEN = 768
    special_tokens = {"bos_token": "<OVERV_START>",
                      "eos_token": "<OVERV_END>",
                      "unk_token": "<|UNK|>",
                      "pad_token": "<|PAD|>",
                      "additional_special_tokens": [
                          "<NAME_START>",
                          "<NAME_END>",
                          "<DESCR_START>",
                          "<DESCR_END>",
                          "<FEAT_START>",
                          "<NEXT_FEAT>",
                          "<FEAT_END>"]}

    # define tokenizer and load the fine-tuned GPT
    tokenizer = myTokenizer(MODEL_NAME, special_tokens)
    model = AutoModelWithLMHead.from_pretrained(
        '../model_medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to("cuda:0")  # gpu usage

    T = Tester(test_df, model, tokenizer)
    new_df = T.generate()

    print(new_df.mean())

    # for graph visualizations
    # BLEU graph
    # axes = new_df.hist(column="bleu_mean")
    # for ax in axes.flatten():
    #     ax.set_ylabel("Count")
    #     ax.set_xlabel("BLEU")

    # GLEU graph
    # axes = new_df.hist(column="gleu_mean")
    # for ax in axes.flatten():
    #     ax.set_ylabel("Count")
    #     ax.set_xlabel("GLEU")