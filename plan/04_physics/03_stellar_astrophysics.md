# Yıldız astrofiziği

Hedef modül: `yildiz/`.

Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. Gözlenebilirlerden fiziksel niceliklere

- Paralakstan uzaklık: `d = 1/ϖ` yalnızca küçük göreli hata için kullanılabilir; genel durumda sonsal dağılım [BailerJones2015; BailerJones2021]. Gaia paralaks sıfır noktası düzeltmesi kadire, renge ve konuma bağlıdır [Lindegren2021].
- Işınım gücü: `L = 4π R² σ T_eff⁴`; bolometrik düzeltme.
- Renk-sıcaklık bağıntıları ve tayf türü sınıflaması [CarrollOstlie2017, Böl. 8].
- Çift yıldızlardan kütle: görsel, tayfsal ve örten çiftler [CarrollOstlie2017, Böl. 7].
- Kütle-ışınım gücü bağıntısı ve anakol ömrü.

## 2. Hertzsprung-Russell diyagramı

- Renk-kadir diyagramının Gaia verisinden kurulması: kalite kesimleri, paralaks hatası kesimi, sönümleme düzeltmesi.
- Evrim bölgeleri: anakol, alt dev kolu, kırmızı dev kolu ve ucu (TRGB), yatay kol, asimptotik dev kolu, beyaz cüce dizisi.
- TRGB'nin standart mum olarak kullanımı: helyum parlamasının I bandındaki sabit ışınım gücü [Freedman2019].

## 3. Yıldız yapısı denklemleri

Küresel simetrik, hidrostatik dengedeki yıldız için [Kippenhahn2012, Böl. 1-10]:

```
dm/dr = 4π r² ρ                       kütle sürekliliği
dP/dr = −G m ρ / r²                   hidrostatik denge
dL/dr = 4π r² ρ ε                     enerji üretimi
dT/dr = −3 κ ρ L / (16π a c r² T³)    ışınımla taşınım
dT/dr = (1 − 1/γ) (T/P) dP/dr         adyabatik konveksiyon
```

Kapanış bağıntıları: durum denklemi `P(ρ, T, X)`, donukluk `κ(ρ, T, X)`, enerji üretim hızı `ε(ρ, T, X)`. Konveksiyon ölçütü Schwarzschild ölçütüdür.

- **Politroplar:** `P = K ρ^(1+1/n)` ve Lane-Emden denklemi. `n = 0, 1, 5` için kapalı çözüm vardır ve sayısal çözücünün testidir; `n = 1,5` tam konvektif yıldızı, `n = 3` Eddington standart modelini verir [Kippenhahn2012, Böl. 19].
- **Durum denklemi:** ideal gaz, ışınım basıncı, yozlaşmış elektron gazı (görelilikli olmayan ve aşırı görelilikli limitler); Chandrasekhar kütlesi [CarrollOstlie2017, Böl. 16].
- **Donukluk:** Kramers yaklaşımı, elektron saçılması; tablolanmış donukluklar ileri aşamada.
- **Çekirdek tepkimeleri:** pp zinciri, CNO döngüsü, üçlü alfa; sıcaklık duyarlılığı için kuvvet yasası yaklaşımları [Kippenhahn2012, Böl. 18].
- **Basit model:** homojen bileşimli sıfır yaş anakol yıldızı için atış yöntemiyle dört denklemin çözümü. Tam evrim kodu yazılmaz; MESA çıktıları karşılaştırma verisidir [Paxton2011].

## 4. İzokronlar ve yıldız popülasyonları

- Evrim izleri ve izokron ızgaraları: MIST [Choi2016] ve PARSEC [Bressan2012]. Izgaralar indirilir, okunur ve yaş-metallik uzayında enterpole edilir.
- Küme uydurma: renk-kadir diyagramında yaş, metallik, uzaklık modülü ve kızarmanın birlikte kestirimi; olabilirlik izokrona uzaklık ve başlangıç kütle fonksiyonu ağırlığıyla kurulur.
- Başlangıç kütle fonksiyonu: Salpeter kuvvet yasası [Salpeter1955], parçalı kuvvet yasası [Kroupa2001], log-normal biçim [Chabrier2003]; normalleştirme ve örnekleme.
- Basit yıldız popülasyonu sentezi: izokron + kütle fonksiyonu + tayf kitaplığı (ileri aşama).

## 5. Değişen yıldızlar

- Zonklayan değişenler: Sefeidler, RR Lyrae, Mira; kararsızlık şeridi.
- **Leavitt yasası** (dönem-ışınım gücü bağıntısı): `M = α log10 P + β` [LeavittPickering1912]; Wesenheit biçimi kızarmayı dışarıda bırakır [Madore1982]; metallik bağımlılığı uzaklık merdiveninin tartışmalı terimlerindendir [Riess2022].
- Örten çiftler: ışık eğrisi geometrisi, Roche modeli (ileri aşama).
- Dönem bulma yöntemleri [../05_veri_bilimi/04_zaman_serisi.md](../05_veri_bilimi/04_zaman_serisi.md) dosyasında.

## 6. Yıldız atmosferleri ve tayf

- Gri atmosfer ve Eddington yaklaşımı: `T⁴(τ) = (3/4) T_eff⁴ (τ + 2/3)` [CarrollOstlie2017, Böl. 9].
- Çizgi oluşumu, eşdeğer genişlik, büyüme eğrisi; bolluk çözümlemesinin temeli.
- Dikine hız: çapraz ilinti [TonryDavis1979]; barisentrik düzeltme [WrightEastman2014].
- Model atmosfer ızgaraları ve sentetik tayf kitaplıkları yeniden üretilmez; okunur ve enterpole edilir.

## 7. Son evreler

Beyaz cüce kütle-yarıçap bağıntısı, nötron yıldızı ve Tolman-Oppenheimer-Volkoff denklemi, süpernova türleri [CarrollOstlie2017, Böl. 15-16]. Tür Ia süpernovaların standartlaştırılması [05_kozmoloji.md](05_kozmoloji.md) dosyasındadır.

## Test durumları

- Lane-Emden: `n = 0, 1, 5` kapalı çözümleri; `n = 3` için ilk sıfırın ve merkez yoğunlaşmasının yayımlanmış tablo değerleri [Kippenhahn2012, Böl. 19].
- Güneş için hidrostatik denge mertebe kestirimleri [CarrollOstlie2017, Böl. 10].
- Kütle fonksiyonu: normalleştirme integrali 1; örneklenen dağılımın KS testi.
- İzokron okuma: MIST ızgarasının bir noktasının enterpolasyonla geri kazanımı.
