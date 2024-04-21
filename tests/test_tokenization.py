import unittest
from src.tokenization import tokenize_data
from transformers import BertTokenizer


class TestTokenization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a small piece of text for testing
        cls.test_text = ["This is a test sentence for tokenization."]
        cls.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def test_tokenization_output(self):
        # Tokenize the test text
        tokenized_output = self.tokenizer(
            self.test_text,
            return_tensors='pt'
        )
        # Check if tokenized output is as expected
        self.assertIsNotNone(tokenized_output['input_ids'])
        self.assertIsNotNone(tokenized_output['attention_mask'])
        self.assertEqual(tokenized_output['input_ids'].shape[1], self.tokenizer.model_max_length)


if __name__ == '__main__':
    unittest.main()
