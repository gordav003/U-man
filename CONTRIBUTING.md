# Prispevanje in organizacija repozitorija

Ta dokument določa preprost način dela, da `main` ostane stabilen in da je iz
zgodovine jasno, zakaj je bila posamezna sprememba narejena.

## Veje

`main` je edina trajna veja in mora vedno vsebovati delujočo, pregledano kodo.
Za vsako zaključeno nalogo ustvari kratkotrajno vejo iz najnovejšega `main`:

- `feature/<kratek-opis>` za novo funkcionalnost ali analizo;
- `fix/<kratek-opis>` za popravek napake;
- `docs/<kratek-opis>` za dokumentacijo;
- `refactor/<kratek-opis>` za preureditev brez spremembe obnašanja;
- `chore/<kratek-opis>` za odvisnosti in vzdrževalna opravila;
- `agent/<kratek-opis>` za spremembe, ki jih pripravi Codex.

Uporabljaj male črke, angleške izraze in vezaje, na primer
`feature/annual-voltage-duration`. Po združitvi PR-ja vejo izbriši lokalno in
na GitHubu.

## Običajen potek dela

1. Posodobi `main` in iz njega ustvari namensko vejo.
2. V en commit združi eno smiselno spremembo; večje naloge lahko imajo več
   ločenih commitov.
3. Pred objavo zaženi ustrezna preverjanja.
4. Odpri pull request proti `main` in v opisu navedi namen, vpliv ter izvedena
   preverjanja.
5. Po pregledu združi PR in izbriši zaključeno vejo.

Ne commitaj merilnih podatkov, poverilnic, lokalnih rezultatov ali virtualnih
okolij. Ti artefakti morajo ostati pokriti z `.gitignore`.

## Commiti in pull requesti

Naslov naj bo kratek in v velelnem naklonu, na primer:

- `Add annual voltage duration analysis`
- `Fix missing segment boundaries`
- `Document Parquet smoke test`

PR naj odgovori na štiri vprašanja: kaj se spreminja, zakaj, kakšen je vpliv
na uporabnika in kako je bila sprememba preverjena.

## Osnovna preverjanja

Za spremembe Python kode najmanj preveri sintakso:

```powershell
python -m compileall -q .
```

Za spremembe prikazovalnika Parquet zaženi tudi:

```powershell
python parquet_plotter.py --smoke-test
```

Analize, ki potrebujejo merilne datoteke, preveri na omejenem časovnem izseku.
V PR zapiši uporabljeni ukaz in rezultat, vendar ne objavljaj zaupnih podatkov.

## Pregled strukture

- `README.md` opisuje namen projekta, namestitev in uporabo analiz.
- `requirements.txt` je enoten seznam Python odvisnosti.
- `voltage/voltage_data.py` vsebuje skupno odkrivanje in branje napetostnih podatkov.
- `continuous_segments.py` vsebuje skupna pravila za zvezne časovne segmente.
- Analitične skripte so razdeljene v pakete `correlations`, `measurements`,
  `reactive_power` in `voltage`; novo skupno logiko izloči v ponovno uporaben modul.
