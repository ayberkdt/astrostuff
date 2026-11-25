# Diğer uçtan uca projects

Her proje bir modül grubunun kabul sınamasıdır: bilinen bir sonucu açık veriden ya da kendi gözlemimizden yeniden üretir. Sıra yol haritasındaki aşamaları izler.

## P1. Gözlem gecesi planlayıcı

- Girdi: gözlemci konumu, tarih, hedef listesi.
- Çıktı: alacakaranlık saatleri, hedeflerin yükseklik ve hava kütlesi eğrileri, Ay'ın konumu ve uzaklığı, gözlem sırası önerisi.
- Modüller: `zaman`, `koordinat`, `efemeris` (Düzey 1), `gozlem/planlama`.
- Kabul: doğuş-batış ve meridyen geçiş zamanları `astroplan` ve Skyfield ile 1 dakika içinde uyuşur [Morris2018; Meeus1998].

## P2. Sentetik CCD ve indirgeme hattı

- Bilinen parametrelerle ham kare üretimi; bias, karanlık, düz alan ve bilim karelerinden indirgenmiş görüntü.
- Modüller: `gozlem/dedektor`, `indirgeme`, `io/fits`.
- Kabul: artıklar gürültü modeliyle tutarlıdır; foton aktarım eğrisi girilen kazancı ve okuma gürültüsünü geri verir; gerçek ham karelerde sonuç `ccdproc` çıktısıyla piksel düzeyinde uyuşur [Howell2006; Janesick2007].

## P3. Açık küme renk-kadir diyagramı

- İki bantta küme görüntüleri (kendi gözlemimiz ya da arşiv), fotometri, Gaia ile astrometrik ve fotometrik kalibrasyon, öz hareket ve paralaksla üyelik, izokron uydurma.
- Modüller: `fotometri`, `astrometri`, `arsiv/gaia`, `yildiz/izokron`, `istatistik`.
- Kabul: uzaklık modülü, kızarma ve yaş yayımlanmış değerlerle belirsizlik içinde uyuşur; uzaklık Gaia paralaks ortalamasıyla tutarlıdır [Vallenari2023; Choi2016; Lindegren2021].

## P4. Gaia HR diyagramı ve yerel yıldız popülasyonu

- 100 pc içindeki yıldızların ADQL ile seçimi, kalite kesimleri, mutlak kadir ve renk; anakol, dev kolu ve beyaz cüce dizisi.
- Modüller: `arsiv/tap`, `arsiv/gaia`, `yildiz/hr`.
- Kabul: Gaia Collaboration'ın HR diyagramının yapısı yeniden elde edilir; `1/ϖ` ile sonsal uzaklık kestirimi arasındaki fark göreli paralaks hatasına karşı gösterilir [BailerJones2015; Luri2018].

## P5. Ötegezegen geçişi

- TESS ışık eğrisinin indirilmesi, eğilim giderme, BLS ve TLS ile dönem arama, geçiş modeli uydurma, MCMC ile parametre belirsizlikleri.
- Modüller: `arsiv` (MAST), `zamanserisi`, `efemeris` (geçiş modeli), `istatistik/mcmc`.
- Kabul: dönem, derinlik ve süre NASA Exoplanet Archive değerleriyle belirsizlik içinde uyuşur; enjeksiyon-geri kazanım haritası üretilir [Kovacs2002; HippkeHeller2019; MandelAgol2002; Ricker2015].
- Uzantı: aynı geçişin yer tabanlı gözlemi ve diferansiyel fotometrisi.

## P6. Sefeid dönem-ışınım gücü bağıntısı

- Büyük Macellan Bulutu Sefeidlerinin ışık eğrileri (OGLE ya da Gaia), Lomb-Scargle ile dönem, ortalama kadir, Wesenheit kadiri, iki eksende hatalı doğrusal uyum.
- Modüller: `zamanserisi/periyodogram`, `isinim/sonumleme`, `istatistik/uyum`.
- Kabul: eğim ve saçılma yayımlanmış bağıntıyla uyuşur; bağıntı başka bir gökadaya uygulanıp uzaklık modülü kestirilir [LeavittPickering1912; Madore1982; Kelly2007]. Bu proje Hubble projesinin uzaklık merdiveni basamağına bağlanır.

## P7. Gökada tayfından kırmızıya kayma

- SDSS tayflarının indirilmesi, süreklilik normalleştirme, salma ve soğurma çizgilerinin saptanması, şablon çapraz ilintisi.
- Modüller: `arsiv`, `tayf`.
- Kabul: ölçülen kırmızıya kaymalar SDSS katalog değerleriyle belirsizlik içinde uyuşur [TonryDavis1979; York2000].
- Uzantı: kendi uzun yarık tayfımızın ham kareden indirgenmesi [Horne1986].

## P8. Gökada dönme eğrisi

- Yayımlanmış bir dönme eğrisinin disk, gaz ve NFW hale bileşenlerine ayrılması; kütle-ışınım gücü oranı ile hale parametrelerinin dejenerasyonu.
- Modüller: `gokada/potansiyel`, `gokada/donme_egrisi`, `istatistik/mcmc`.
- Kabul: hale parametreleri yayımlanmış uyumla tutarlıdır; karanlık maddesiz model verilerle bağdaşmaz [NFW1997; BinneyTremaine2008].

## P9. Küçük cisim astrometrisi ve yörünge

- Bir asteroitin birkaç gecelik gözleminden astrometri, Gauss yöntemiyle başlangıç yörüngesi, diferansiyel düzeltme.
- Modüller: `astrometri`, `efemeris`.
- Kabul: elemanlar JPL değerleriyle uyuşur; öngörülen konum sonraki gözlemle doğrulanır [Vallado2022; Lang2010].

## P10. Güneş sistemi integrasyonu

- Gezegenlerin DE440 başlangıç koşullarından simplektik integrasyonu; Merkür'ün günberi ilerlemesi (Newton ve post-Newton).
- Modüller: `efemeris`, `temel/sayisal`.
- Kabul: 100 yıllık konumlar DE440 ile uyuşur; enerji hatası sınırlı kalır; REBOUND ile aynı başlangıç koşullarında uyum [Park2021; ReinTamayo2015; WisdomHolman1991].

## P11. Yıldız yapısı: politrop ve basit anakol modeli

- Lane-Emden çözümleri; homojen bileşimli anakol yıldızının atış yöntemiyle çözümü; kütle-ışınım gücü ve kütle-yarıçap bağıntıları.
- Modüller: `yildiz/yapi`.
- Kabul: Lane-Emden kapalı çözümleri; Güneş kütleli modelin merkez koşulları ders kitabı değerleriyle mertebe olarak uyuşur; eğilimler MIST izleriyle tutarlıdır [Kippenhahn2012; Choi2016].

## İleri aşama adayları

Çekimsel dalga olayının eşleşmiş süzgeçle bulunması (GW150914 açık verisi) [Abbott2016]; görüntü farkıyla geçici kaynak arama [Zackay2016]; tayfsal enerji dağılımı uydurma; gökada iki nokta ilinti fonksiyonu ve BAO tepesi [Eisenstein2005]; mercek zaman gecikmesiyle `H0`. Her biri başlamadan önce kendi saha taraması yapılır.
