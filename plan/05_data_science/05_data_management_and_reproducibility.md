# Veri yönetimi ve tekrarlanabilirlik

Bir sonucun, onu üreten veri, kod sürümü ve parametrelerle birlikte yeniden elde edilebilmesi bu projenin kabul ölçütüdür. İlkeler bilimsel hesaplamada yeterince iyi uygulamalar derlemesini ve FAIR ilkelerini izler [Wilson2017; Wilkinson2016].

## 1. Veri katmanları

- `data/raw/`: arşivden indirilen ya da teleskoptan gelen dosyalar. Salt okunur; hiçbir kod üzerine yazmaz. Sürüm denetimi dışındadır; her dosyanın SHA-256 özeti ve kaynağı bir kayıt dosyasında tutulur.
- `data/interim/`: indirgenmiş ama henüz bilim ürünü olmayan dosyalar. Silinip yeniden üretilebilir.
- `veri/urun/`: kataloglar, ışık eğrileri, tayflar, uyum sonuçları. Her ürünün yanında onu üreten çalıştırmanın kaydı durur.
- `veri/referans/`: testlerin kullandığı küçük dosyalar; sürüm denetimindedir.

## 2. Çalıştırma kaydı

Her analiz çalıştırması bir kayıt üretir: kod sürümü (git özeti, kirli çalışma ağacı uyarısı), girdi dosyalarının özetleri, bütün parametreler, rastgele tohum, bağımlılık sürümleri, başlangıç ve bitiş zamanı, çıktı dosyalarının özetleri. Kayıt çıktının yanına JSON olarak yazılır.

Kural: kayıt önce yazılır, ara dosyalar sonra temizlenir. Bir ürünün üzerine yazılmadan önce eski sürümünün özeti kayda geçer.

## 3. Yapılandırma

- Parametreler kodda değil, yapılandırma dosyasındadır (TOML); kod varsayılanı figures, dosya çalıştırmayı tanımlar.
- Aygıt tanımları (kazanç, okuma gürültüsü, piksel ölçeği, başlık anahtarı eşlemesi) ayrı dosyalardır.
- Sihirli sayı yoktur: eşik, pencere ve tolerans değerlerinin her biri adlandırılır ve gerekçesi yazılır.

## 4. İş akışı

- Uzun işlem adımlara bölünür; her adım girdisinin özeti değişmediyse yeniden koşmaz.
- **Pahalı iş öncesi kapı:** saatler süren bir çalıştırma (büyük MCMC, toplu indirgeme, büyük indirme) başlamadan önce küçük bir denetim betiği işin gerçekten gerekli olduğunu veriden doğrular: çıktı zaten var mı, girdi değişti mi, kısa bir deneme koşusu beklenen biçimde sonuç veriyor mu. Denetim çıktı üretir; çıkış kodu sıfır değilse iş başlamaz.
- Uzun çalıştırmalar ara durum kaydeder ve kaldığı yerden sürer; zincirler diske akıtılır.
- Önce küçük ölçekli deneme (kısa zincir, veri alt kümesi), sonra tam çalıştırma.

## 5. Ortam

- Bağımlılıklar `pyproject.toml` içinde alt sınırla, kilit dosyasında tam sürümle tanımlanır.
- Kâhin kütüphanelerinin sürümü referans çıktı dosyalarının başlığına yazılır; kâhin güncellendiğinde fark raporu üretilir.
- Sayısal sonuçlar işletim sistemi ve BLAS uygulamasına son basamaklarda bağlı olabilir; tolerans buna göre konur.

## 6. Büyük veri

- Bellek dışı çalışma: bellek eşlemeli FITS, parça parça okuma, Parquet ve tembel sorgular.
- Gökyüzü bölümleme: HEALPix hücreleriyle paralel işleme ve çapraz eşleme.
- Arşiv tarafında hesap: mümkün olan süzme ADQL sorgusunda yapılır, yalnızca gereken satır ve sütun indirilir.
- Hızlandırma sırası: önce doğru ve vektörleştirilmiş NumPy, ölçüm (profil çıkarma), sonra gerekirse Numba ya da JAX. Ölçülmeden iyileştirme yapılmaz.

## 7. Raporlama

- Her sonuç değeri kaynağıyla bildirilir: hangi dosyadan, hangi çalıştırmadan okunduğu.
- Belirsizlik her zaman değerle birlikte verilir; istatistik ve sistematik ayrı.
- Şekiller betikle üretilir; şeklin verisi ayrıca kaydedilir. Şekil ve tablo üretimi projenin şekil standardına (dergi profili, renk körlüğü güvenli palet, doğrudan etiket) uyar.
- Olumsuz sonuçlar ve başarısız denemeler de kaydedilir.

## 8. Atıf ve lisans

- Kullanılan her veri setinin atıf koşulu `arsiv` modülünde veriyle birlikte tutulur ve analiz raporuna otomatik eklenir.
- `kozmos.bibliography_raporu()` bir analizde çağrılan fonksiyonların dayandığı yayınları listeler.
- Kâhin olarak kullanılan yazılımlar da anılır [Astropy2013; Astropy2018; Astropy2022; Harris2020; Virtanen2020].
