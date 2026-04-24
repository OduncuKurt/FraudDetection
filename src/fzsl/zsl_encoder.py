"""
zsl_encoder.py
--------------
Sinif aciklamalarini embedding vektorlerine donusturen metin encoder.

Backend secenekleri:
  "local" (varsayilan): TF-IDF + SparseRandomProjection
                        - Internet gerektirmez
                        - Belge sayisindan bagimsiz, her zaman tam embed_dim boyutu uretir
                        - Akademik ZSL literaturunde BOW/LSA yontemi yaygindir
  "sbert"             : Sentence-BERT (internet + model download gerektirir)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import normalize


class TextClassEncoder:
    """
    Sinif aciklamalarini sabit boyutlu embedding vektorlerine donusturmek icin
    TF-IDF + SparseRandomProjection kullanan encoder.

    SparseRandomProjection kullanilmasinin nedeni:
      - SVD'nin aksine, belge sayisindan bagimsiz calisir
      - 5 sinif aciklamasi olsa da tam embed_dim boyutu garantilenir
      - Johnson-Lindenstrauss Lemma'ya gore uzaklik iliskilerini korur
    """

    def __init__(self, embed_dim: int = 128, backend: str = "local",
                 sbert_model: str = "all-MiniLM-L6-v2"):
        self.embed_dim = embed_dim
        self.backend = backend
        self._fitted = False

        if backend == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[TextClassEncoder] SBERT yukleniyor: {sbert_model}")
                self._sbert = SentenceTransformer(sbert_model)
                self.embed_dim = self._sbert.get_sentence_embedding_dimension()
                print(f"[TextClassEncoder] SBERT embed_dim: {self.embed_dim}")
            except Exception as e:
                print(f"[TextClassEncoder] SBERT yuklenemedi ({e}). 'local' backend'e geciliyor.")
                self.backend = "local"

        if self.backend == "local":
            # Karakter n-gram: kisa metinlerden bile zengin ozellik cikarir
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",        # karakter n-gram
                ngram_range=(3, 6),        # 3'ten 6'ya karakter dizileri
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode",
                lowercase=True,
            )
            # Rastgele projeksiyon: her zaman embed_dim cikti garantisi
            self._projector = SparseRandomProjection(
                n_components=embed_dim,
                random_state=42,
                density="auto",
            )
            print(f"[TextClassEncoder] Local TF-IDF+RandomProjection hazir (dim={embed_dim})")

    def _fit_local(self, texts: list):
        """TF-IDF + RandomProjection'i sinif aciklamalarina gore fit eder."""
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        # SparseRandomProjection belge sayisindan bagimsiz calisir
        self._projector.fit(tfidf_matrix)
        self._fitted = True
        print(f"[TextClassEncoder] TF-IDF vocab boyutu: {len(self._vectorizer.vocabulary_)}")

    def _encode_local(self, texts: list) -> np.ndarray:
        tfidf_matrix = self._vectorizer.transform(texts)
        proj_out = self._projector.transform(tfidf_matrix)
        return normalize(proj_out.toarray() if hasattr(proj_out, 'toarray') else proj_out,
                         norm="l2")

    def encode_descriptions(self, class_descriptions: dict) -> dict:
        """
        class_descriptions: {class_name: description_text}
        Donus: {class_name: L2-normalize edilmis embedding (np.array)}
        """
        class_names = list(class_descriptions.keys())
        texts = [class_descriptions[name] for name in class_names]

        if self.backend == "sbert":
            embeddings = self._sbert.encode(texts, normalize_embeddings=True,
                                            show_progress_bar=False)
        else:
            if not self._fitted:
                self._fit_local(texts)
            embeddings = self._encode_local(texts)

        return {name: emb for name, emb in zip(class_names, embeddings)}

    def get_class_embedding_matrix(
        self, class_descriptions: dict, class_order: list = None
    ):
        """
        Belirli bir sirayla sinif embedding matrisini dondurur.

        Donus:
            matrix     : np.array, shape=[num_classes, embed_dim]
            class_order: sinif isimlerinin sirasi
        """
        if class_order is None:
            class_order = list(class_descriptions.keys())

        emb_dict = self.encode_descriptions(class_descriptions)
        matrix = np.stack([emb_dict[cls] for cls in class_order])
        actual_dim = matrix.shape[1]
        print(f"[TextClassEncoder] Sinif embedding matrisi: {matrix.shape}  "
              f"(hedef={self.embed_dim}, gercek={actual_dim})")
        return matrix, class_order
