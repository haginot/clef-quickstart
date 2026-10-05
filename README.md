# Clef Quickstart

Cloudflare のオープンな意思決定モデル [Clef](https://huggingface.co/Cloudflare/clef) / [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) をローカルで検証するための最小構成です。

Clef はチャット文章を生成するモデルではありません。`state`（判断材料）と `questions`（型付きの判断項目）を受け取り、許可された選択肢ごとの確率を返します。自由文のパースが不要なので、分類、ルーティング、スコアリング、ワークフローの分岐に向いています。

## モデルの選び方

| モデル | 規模 | 向いている用途 |
|---|---:|---|
| `clef-flash` | 9B | 最初の検証、低遅延が重要な処理 |
| `clef` | 27B | 品質を優先する検証 |

どちらもテキスト、JSON、画像、動画を入力でき、Apache-2.0 で公開されています。公式カードの検証環境は PyTorch 2.11、Transformers 5.10.2、単一 H200 です。このリポジトリではダウンロード量と計算資源を抑えるため `clef-flash` を既定にしています。

## すぐ試す

前提は Python 3.11+ と、BF16 を扱える十分なメモリを持つ NVIDIA GPU です。モデルの重みは初回実行時に Hugging Face のキャッシュへダウンロードされます。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# モデルを取得せず、入力形式だけ確認
python clef_quickstart.py examples/incident.json --validate-only

# Clef-Flash で推論
python clef_quickstart.py examples/incident.json
```

大きいモデルを試す場合は、入力 JSON の `model` を `clef` に変更します。GPU 以外での実行は公式の検証対象外です。`--device cpu` も指定できますが、BF16 の対応状況、メモリ、実行時間に注意してください。

## 入力形式

```json
{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {
        "billing": "Payments or invoices",
        "technical": "Bugs or outages"
      }
    },
    "urgency": {
      "type": "score",
      "criteria": ["Can wait", "This week", "Today"]
    },
    "outage": {
      "type": "noul",
      "instructions": "Is a service down?"
    }
  }
}
```

質問の型は次の3種類です。

- `noul`: true / false の確率
- `choice`: `criteria` で定義した選択肢ごとの確率と最有力候補
- `score`: 順序付きレベルごとの確率と期待スコア

出力は Jev / SystemOne の `POST /v1/systemone` と互換で、`answers` に質問ごとの判断、信頼度、確率が入ります。

## 画像・動画

公式 API は `images` に PIL Image、`videos` にフレーム配列を渡せます。ただし JSON ファイルからはそのまま表せないため、この最小 CLI はまずテキスト / JSON 状態の検証に絞っています。マルチモーダル検証では公式モデルカードの `encode_record(..., processor=processor)` の例を参照してください。

## 開発時の確認

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## 調査メモ

- Backbone は `clef` が Qwen3.8-27B、`clef-flash` が Qwen3.5-9B。
- 独自の joint schema head が全質問と全選択肢を同時に評価し、質問ごとに softmax した確率を返す。
- 最大入力長の既定値は 16,384 tokens。
- Hugging Face Inference Providers には現時点で未配備。公式利用例はローカルで `snapshot_download` し、配布物内の `joint_schema_model.py` を使う方法。
- `pipeline("image-text-to-text")` の一般例も表示されるが、Clef 固有の型付き意思決定には `systemone` または `encode_record` / `collate_records` を使う。

## 参照

- [Cloudflare/clef](https://huggingface.co/Cloudflare/clef)
- [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash)
- [Cloudflare 公式ブログ: Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
- [Decision Index](https://clef-evals.workers-ai-mle.workers.dev/)

## License

このサンプルコードは MIT License で公開します。Clef のモデル本体は Apache-2.0 です。
