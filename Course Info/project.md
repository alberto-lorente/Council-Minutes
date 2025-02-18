# Projet Datapolitics

L'objet du projet est de réaliser un détecteur de "projets" mis en œuvre par les collectivités territoriales à partir
des documents qu'elles produisent.

On se concentre ici sur la géothermie.

Le matériau est un ensemble (environ 20.000) de documents PDF produits depuis 5 ans par ces collectivités et mis en 
ligne sur
leur site. Il s'agit la plupart du temps de compte-rendus de conseils municipaux ou régionaux, mais le type de document
n'est pas filtré a priori. Les documents ont été extraits de la base Datapolitics sur la présence du mot-clé
"géothermie" (qui est peu ambigu).

Comme référence des projets actuellement mis en œuvre et pour se familiariser avec le domaine, on pourra consulter le
site (géothermies)[https://www.geothermies.fr]. Le site mentionne en particulier les acronymes des projets ou
installations, ce qui peut se révéler précieux dans une phase d'exploration pour retrouver leur trace dans les 
documents.

L'objectif est de fournir (automatiquement), pour chacun des documents, son statut par rapport à un projet de
géothermie :

A un premier niveau, on demande un filtre binaire sur les documents:

- concerne un projet de géothermie
- rien à voir avec un projet de géothermie

A un second niveau, pour ceux qui passent le premier filtre, on distingue les étapes suivantes:

- idée/souhait,
- études préalables,
- budget voté pour le projet définitif,
- réalisation en cours,
- installation terminée

Dernier niveau d'extraction: les données du projet:

- budget initial
- coût final
- durée estimée
- durée effective

Le rendu consistera en une chaîne de traitement permettant le filtrage et l'extraction des informations. La forme n'est
pas contrainte (scripts, Notebooks, etc.) sont
On s'attachera à ce que la méthodologie mise en œuvre puisse être appliquée à tous types de projets locaux (éolien,
construction de maison de santé, rénovation de l'éclairage public, etc.)

# Jeu de données

Le jeu de données est décrit par le fichier `dataset.csv`. Les URLs renvoient à la version source des documents mais
ils sont tous disponibles en cache. Une version convertie en texte plein est également disponible.

Le schéma du fichier est le suivant

| Champ       | Contenu                                                     |
|-------------|-------------------------------------------------------------|
| doc_id      | Identifiant (interne Datapolitics) du document              |
| url         | url source                                                  |
| cache       | lien vers la version en cache du PDF                        |
| fulltext    | lien vers la version convertie en texte                     |
| nature      | Type de document (calculé automatiquement)                  |
| published   | Date de publication (caclulée automatiquement)              |
| entity_name | Nom de l'entité locale                                      |
| entity_type | type d'entité (commune, intercommunalité, préfecture, etc.) |
| geo_path    | "chemin" dans la structure administrative                   |

Le champ `geo_path` permet de "remonter" les niveaux administratifs. Ces niveaux sont séparés par des "/". Typiquement,
pour une commune, ce sera : `nom de la commune/nom de l'intercommunalité/nom du département/nom de la région`. Pour
des niveaux plus hauts, le chemin sera plus court.

Les types de documents sont produits par un classifieur automatique. La nomenclature est la suivante :

| Code               | Signification                       |
|--------------------|-------------------------------------|
| acte.delib         | Délibération                        |
| acte.arrete        | Arrêté                              |
| acte.raa           | Recueil d'actes administratifs      |
| pv.full            | Compte rendu                        |
| pv.cr              | Compte rendu                        |
| pv.odj             | Ordre du jour                       |
| pv.video           | Vidéo                               |
| projet             | Projet d'acte                       |
| dlao.plu           | Plan local d'urbanisme              |
| dlao.pcaet         | Plan climat-air-énergie territorial |
| dlao.scot          | Schéma de cohérence territoriale    |
| dlao.autres        | Autre document de planification     |
| mp.annonce         | Annonce de marché public            |
| mp.avenant         | Avenant au marché public            |
| mp.reglement       | Règlement des marchés publics       |
| comm               | Autre communication                 |
| comm.org           | Organigramme                        |
| rapport            | Rapport                             |
| bdj                | Budget                              |
| bdj.annexes        | Document budgétaire annexé          |
| comm.agenda        | Agenda                              |
| comm.contact       | Contact                             |
| comm.concert       | Concertation                        |
| comm.demarches     | Démarches                           |
| comm.emploi        | Emploi                              |
| comm.present       | Présentation                        |
| comm.project       | Projet                              |
| comm.magazine      | Périodique                          |
| comm.election      | Elections                           |
| comm.services      | Services                            |
| other              | Autre                               |
