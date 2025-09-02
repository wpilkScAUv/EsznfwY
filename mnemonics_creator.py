"""AnkiMnemonics module for generating mnemonic devices for Anki flashcards.

This module provides the AnkiMnemonics class which integrates with Anki via AnkiConnect
to help users create memorable mnemonics for difficult cards. It uses LLMs to generate
contextually relevant mnemonics based on card content and optional memory anchors.

Key features:
- Identifies cards that need mnemonics based on review history and tags
- Uses LLMs (like GPT-4, Claude) to generate mnemonics
- Preserves history of previous mnemonics
- Supports semantic filtering and memory anchors
- Integrates with ntfy.sh for mobile notifications
- Tracks token usage and costs

The main class AnkiMnemonics handles:
- Card selection and filtering
- LLM prompt construction and response handling  
- Anki card updates
- History management
- Cost tracking
"""
from inspect import signature
from bs4 import BeautifulSoup
from tqdm import tqdm
from tqdm_logger import TqdmLogger
import datetime
import re
import time
import json
from pathlib import Path
from typing import List
import fire
import copy

from utils.misc import send_ntfy, load_formatting_funcs, replace_media
from utils.anki import anki, sync_anki, addtags, removetags, updatenote
from utils.logger import create_loggers
from utils.llm import load_api_keys, llm_price, llm_cost_compute, tkn_len, chat, model_name_matcher
from utils.datasets import load_dataset, load_and_embed_anchors, filter_anchors, semantic_prompt_filtering

Path("databases").mkdir(exist_ok=True)
MNEMONICS_PATH = Path("databases/mnemonics/")
MNEMONICS_PATH.mkdir(exist_ok=True)

MNEMONICS_HIST_PATH = MNEMONICS_PATH / "mnemonics_history.json"
log_file = MNEMONICS_PATH / "mnemonics_logs.txt"
Path(log_file).touch()
whi, yel, red = create_loggers(log_file, ["white", "yellow", "red"])

# load api key
load_api_keys()


class AnkiMnemonics:
    VERSION = "2.1"

    def __init__(
        self,
        query: str = "(rated:2:1 OR rated:2:2 OR tag:AnkiMnemonics::todo OR tag:AnkiMnemonics::failed) -is:suspended -tag:AnkiMnemonics::to_keep",
        # model: str = "openai/gpt-4o",
        # model: str = "anthropic/claude-3-5-sonnet-20240620",
        model: str = "openrouter/anthropic/claude-3.5-sonnet:beta",
        field_names: List[str] = None,
        llm_max_token: int = 3000,
        # embedding_model: str = "mistral/mistral-embed",
        embedding_model: str = "openai/text-embedding-3-small",
        n_mnemonic: int = 1,
        memory_anchors_file: str = None,
        dataset_path: str = None,
        string_formatting: str = None,
        force: bool = False,
        note_mode: bool = True,
        do_sync: bool = True,  # don't sync by default because it can impede card creation otherwise
        ntfy_url: str = None,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        query: str, default "(rated:2:1 OR rated:2:2 OR tag:AnkiMnemonics::todo OR tag:AnkiMnemonics::failed) -is:suspended -tag:AnkiMnemonics::to_keep"
            will have " deck:Externat note:Clozolkor*"
            appended to it automatically

        model: str, default to anthropic/claude-3-5-sonnet-20240620

        field_names: List[str], default None
            list (or comma separated string) of the field of the note to load
            and give to the LLM as prompt.

        llm_max_token: int, default 3000
            max number of token when asking the LLM for a prompt

        embedding_model: str, default "openai/text-embedding-3-small"
            embedding model to use, in litellm format

        note_mode: bool, default True
            if True, then cards of the same note will not be counted twice

        n_mnemonic: int, default 1
            number of mnemonic per card by default. Only supports by openai
            models.

        memory_anchors_file: str

        dataset_path: str
            path to a file with ---- separated messages (including system
            prompt) showing a succession of example

        string_formatting: str, default None
            path to a python file declaring functions to specify specific
            formatting.

            In mnemonics, functions that can be loaded are:
            - "cloze_input_parser"
            it mist take a unique string argument and return a unique string.

            They will be called to modify the note content before sending
            to the LLM

        force: bool, default False
            if True, will run even if the card was previously processed

        do_sync: bool, default True
            if True: will sync anki on start

        ntfy_url: str, default None
            url to use with ntfy.sh to send the status updates

        debug: bool, default False
            if True, will launch a debug console at the end
        """
        # logger for tqdm progress bars
        self.t_strm = TqdmLogger(log_file)
        self.t_strm.reset()
        self.n_mnemonic = n_mnemonic
        self.ntfy_url = ntfy_url

        self.llm_model = model
        self.embedding_model = embedding_model
        if self.llm_model in llm_price:
            self.llm_price = llm_price[self.llm_model]
        elif self.llm_model.split("/", 1)[1] in llm_price:
            self.llm_price = llm_price[self.llm_model.split("/", 1)[1]]
        elif model_name_matcher(self.llm_model) in llm_price:
            self.price = llm_price[model_name_matcher(self.llm_model)]
        else:
            raise Exception(f"{self.llm_model} not found in llm_price")
        self.llm_max_token = llm_max_token

        if not force:
            query += f" -AnkiMnemonics:*VERSION:{self.VERSION}* "
        else:
            red("--force enabled, this will not ignore cards with mnemonics")

        if isinstance(field_names, list):
            assert not any("," in f for f in field_names), (
                "Detected a list of field_names where one contains a comma")
        else:
            assert isinstance(field_names, str)
            field_names = field_names.split(",")
        self.field_names = field_names

        # load user_anchors
        self.anchors = {}
        if memory_anchors_file:
            self.anchors, self.embeddings = load_and_embed_anchors(
                path=memory_anchors_file,
                model=embedding_model,
            )

        if string_formatting is not None:
            red(f"Loading specific string formatting from {string_formatting}")
            cloze_input_parser = load_formatting_funcs(
                    path=string_formatting,
                    func_names=["cloze_input_parser"]
            )[0]
            for func in [cloze_input_parser]:
                params = dict(signature(func).parameters)
                assert len(params.keys()) == 1, f"Expected 1 argument for {func}"
                assert "cloze" in params, f"{func} must have 'cloze' as argument"
            self.cloze_input_parser = cloze_input_parser
        self.string_formatting = string_formatting

        self.dataset = load_dataset(dataset_path)
        # turn easy to type formatting into proper
        for i, m in enumerate(self.dataset):
            if i == 0:
                continue
            if m["role"] == "user":
                if self.anchors:
                    if "Anchors:" not in m["content"]:
                        whi(f"Missing 'Anchors:' in dataset message #{i+1}: '{m['content']}'")
            elif m["role"] == "assistant":
                for line in m["content"].split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    assert (
                        (line.startswith("'") and line.endswith("'"))
                        or (
                            line.startswith("* ")
                            and " => " in line
                            and " ## " in line
                            and line.endswith(" <=")
                            and line.index(" => ") < line.index(" ## ")
                        )
                    ), f"Invalid line in assistant message: '{line}' of '{m['content']}'"
                lines = m["content"].splitlines()
                cont = "\n".join([li.strip() for li in lines if li.strip()])

                cont = cont.replace("* ", "* <b>")
                cont = cont.replace("=>", "</b>")
                cont = cont.replace("##", "<u>")
                cont = cont.replace("<=", "</u>")

                # make sure to add missing formatting markups termination
                lines = cont.splitlines()
                for il, li in enumerate(lines):
                    for symb in ["b", "u"]:
                        li = li + f"</{symb}>" * max(0, li.count(f"<{symb}>") - li.count(f"</{symb}>"))
                    lines[il] = li
                cont = "\n".join(lines)

                self.dataset[i]["content"] = cont

        # sync first
        if do_sync:
            sync_anki()

        # load history of already mnemonized cards
        self._load_history()

        # find cid of recently failed cards
        red(f"Loading failed cards with query '{query}'")
        failed = anki(action="findCards", query=query)
        self.note_mode = note_mode
        if note_mode:
            whi(f"note_mode enabled, don't count cards of the same note twice.")

        if not failed:
            raise SystemExit("No card corresponding to query found")

        yel(f"Found '{len(failed)}' cards failed recently")

        # gather info about those failed cards
        failed_info = anki(action="cardsInfo", cards=failed)
        assert len(failed_info) == len(failed), "Invalid cards info length"
        self.failed_info = failed_info

        # filter cards based on history
        self._filter_failed()

        # abort if more than X cards
        if len(self.failed_info) > 1000:
            raise Exception(
                "too many cards, aborting just in case: " f"'{len(self.failed_info)}'"
            )

        # create mnemonics and send notifications, by deck
        pbar = tqdm(
            total=len(self.failed_info) + len(self.deck_list),
            unit="cards",
            file=self.t_strm,
        )
        cnt = 0
        for deck in self.deck_list:
            cards = [c for c in self.failed_info if c["deckName"] == deck]
            cards = sorted(cards, key=lambda x: x["formatted_content"])
            to_send = []
            for card in cards:
                cnt += 1
                cid = str(card["cardId"])
                content = card["formatted_content"]

                # ask llm
                response = self._create_mnemonics(
                    card_content=content,
                )
                mnemonics = [
                    m["message"]["content"]
                    for m in response["choices"]
                ]
                input_cost = response["usage"]["prompt_tokens"]
                output_cost = response["usage"]["completion_tokens"]

                assert (
                    len(mnemonics) == self.n_mnemonic
                ), f"Created {len(mnemonics)} instead of the asked {self.n_mnemonic}"

                # format mnemonics as one string
                temp = "1.  "
                for ii, mn in enumerate(mnemonics):
                    temp += mn
                    temp += "</b></i></u>"
                    if mn != mnemonics[-1]:
                        temp += f"\n\n{ii + 2}.  "
                mnemonics = temp

                whi(f"Mnemonics for '{content}':\n{mnemonics}\n")

                to_send.append(mnemonics)

                self._edit_anki_card(card=card, mnemonics=mnemonics)

                if cid in self.history:
                    self.history[cid]
                    self.history[cid].append(
                        {
                            "cardsInfo": card,
                            "timestamp": int(time.time()),
                            "datetime": self.today,
                            "mnemonics": mnemonics,
                            "obsolete": False,
                            "input_cost": input_cost,
                            "output_cost": output_cost,
                            "dollar_cost": llm_cost_compute(
                                input_cost, output_cost, self.llm_price
                            ),
                            "input_string": content,
                        }
                    )
                else:
                    self.history[cid] = [
                        {
                            "cardsInfo": card,
                            "timestamp": int(time.time()),
                            "datetime": self.today,
                            "mnemonics": mnemonics,
                            "obsolete": False,
                            "input_cost": input_cost,
                            "output_cost": output_cost,
                            "dollar_cost": llm_cost_compute(
                                input_cost, output_cost, self.llm_price
                            ),
                            "input_string": content,
                        }
                    ]
                self._save_history()

                pbar.update(1)

                # sync regularly
                if cnt % 100 == 0:
                    sync_anki()

            pbar.update(1)
            tqdm.write(f"Done with deck '{deck}\n\n'")

            self._send_notif(contents=to_send, deckname=deck)

        pbar.close()

        # add and remove the tag TODO to make it easier to readd by the user
        # as it was cleared by calling 'clearUnusedTags'
        addtags(card["note"], tags="AnkiMnemonics::TODO")
        removetags(card["note"], tags="AnkiMnemonics::TODO")

        # sync at the end
        if do_sync:
            sync_anki()

        if debug:
            red("Finished. Openning console.")
            breakpoint()
        else:
            red("Finished.")
            raise SystemExit()

    def _edit_anki_card(self, card, mnemonics):
        """Update an Anki card with new mnemonics while preserving history.

        Parameters
        ----------
        card : dict
            Dictionary containing card information from Anki, including:
            - cardId: unique card identifier
            - note: note ID 
            - fields: dict containing card field values

        mnemonics : str
            New mnemonics text to add to the card's AnkiMnemonics field.
            Previous mnemonics will be preserved in a collapsible section.
        """
        whi(f"Editing anki card '{card['cardId']}'")
        nid = int(card["note"])
        previous = card["fields"]["AnkiMnemonics"]["value"].strip()

        # remove old previous field
        if "VERSION:" not in previous:
            previous = ""

        # remove previous detail tag
        previous = re.sub(r"\</?details\>|\</?mnemonics\>",
                          "", previous).strip()

        if previous:
            # wrap the previous content in a detail tag
            previous = (
                f"<details><summary>Previous mnemonics</summary>{previous}</details>"
            )

        new = mnemonics
        new += "<br><br>"
        new += f"[DATE:{self.today} VERSION:{self.VERSION} MODEL:{self.llm_model}]"
        new += "<br><br>"
        new += "<!--SEPARATOR-->"
        new += previous

        new = new.replace("\r", "<br>").replace("\n", "<br>")  # html newlines

        try:
            updatenote(nid, fields={"AnkiMnemonics": new})

            # add tag to note if success
            addtags(nid, tags=f"AnkiMnemonics::done::{self.today}")

            # remove todo and failed tags
            removetags(nid, tags="AnkiMnemonics::failed AnkiMnemonics::todo")

            # remove doall tag if all done
            notetags = anki(
                action="getNoteTags",
                note=int(nid),
            )
            assert isinstance(notetags, list), "notetags is not list"
            tagcheck = [
                f"AnkiMnemonics::done::{self.today}",
                f"AnkiSummary::done::{self.today}",
                f"AnkiIllustrator::done::{self.today}",
            ]

        except Exception as err:
            red(f"Exception when editing '{nid}': '{err}'")
            # add tag to note if faileed
            addtags(nid, tags="AnkiMnemonics::failed")

    def _create_mnemonics(self, card_content):
        """Generate mnemonics for a card using LLM.

        Parameters
        ----------
        card_content : str
            The formatted content of the card to generate mnemonics for,
            including any anchors and field content.

        Returns
        -------
        dict
            Response from the LLM containing:
            - choices: list of generated mnemonics
            - usage: token usage statistics
        """
        messages = semantic_prompt_filtering(
            curr_mess={"role": "user", "content": card_content},
            max_token=self.llm_max_token,
            temperature=0,
            prompt_messages=copy.deepcopy(self.dataset),
            keywords="",
            embedding_model=self.embedding_model,
            whi=whi,
            yel=yel,
            red=red,
        ) + [
                {
                    "role": "user",
                    "content": card_content
                    }
                ]

        whi("Asking LLM")
        assert tkn_len(messages) <= self.llm_max_token
        response = chat(
            messages=messages,
            model=self.llm_model,
            temperature=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            n=self.n_mnemonic,
            num_retries=5
        )
        return response

    def _filter_failed(self):
        """
        removes from the list of failed cards the one that were already
        notified in the recent days

        also removes the cloze indicator of another card but the same note
        (= remove c1 if the failed cards was c2).
        """
        to_filter = []
        nids_so_far = []

        d = datetime.datetime.today()
        if d.hour <= 5:
            # get yesterday's date if it's too early in the day
            d = datetime.datetime.today() - datetime.timedelta(1)
        self.today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

        for i, f in enumerate(tqdm(self.failed_info, file=self.t_strm)):
            cid = str(f["cardId"])

            # don't count cards of the same note type twice
            if self.note_mode:
                nid = f["note"]
                if nid in nids_so_far:
                    to_filter.append(cid)
                else:
                    nids_so_far.append(nid)

            # filter if the card is not a 'relearning' or 'review' card
            # (i.e. exclude new cards)
            # reference: 0=new, 1=learning, 2=review, 3=relearning
            # if int(f["type"]) not in [2, 3]:
            #     to_filter.append(cid)
            #     continue

            fields = f["fields"]
            content = ""
            for fn in self.field_names:
                content += f"\n{fn.title()}: {fields[fn]['value'].strip()}"
            content = content.strip()

            orig_content = content

            content, _ = replace_media(
                content=content,
                media=None,
                mode="remove_media")

            # light formatting
            if self.string_formatting:
                content = self.cloze_input_parser(content)

            content = re.sub("{{c[0-9]+::(.*?)}}", r"\1",
                             content, flags=re.DOTALL | re.M)

            content = content.replace("<br>", "\n").replace("<br/>", "\n")
            content = content.replace("\r", "\n")
            content = content.replace(" }}\n", "}}\n")

            # identify anchors
            if self.anchors:
                matching_anchors = filter_anchors(
                    n=15,
                    content=content,
                    anchors=self.anchors,
                    embeddings=self.embeddings,
                    model=self.embedding_model,
                )

                anchors_to_add = " ; ".join([f"{k.strip()}: {v.strip()}" for k, v in matching_anchors]).strip()

                content += "\n\nAnchors: '" + anchors_to_add + "'"
                whi(f"Anchors: '{anchors_to_add.strip()}'")

            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text()

            yel(f"Old content: '{orig_content}'")
            red(f"New content: '{content}'")
            print("")

            # store content for history
            self.failed_info[i]["formatted_content"] = content

        yel(f"Cards filtered: '{len(to_filter)}'")
        self.failed_info = [
            f for f in self.failed_info if str(f["cardId"]) not in to_filter
        ]
        yel(f"Cards to mnemonize: '{len(self.failed_info)}'")

        if not self.failed_info:
            raise SystemExit("No cards to notify of after filtering.")

        # get list of unique decks
        self.deck_list = list(set([f["deckName"] for f in self.failed_info]))
        self.deck_list = sorted(self.deck_list, reverse=True)
        red(f"Unique decks: '{self.deck_list}'")

    def _send_notif(self, contents, deckname):
        """
        send notification to phone
        """
        if not self.ntfy_url:
            return
        # shorten deckname
        deckname = "::".join(deckname.split("::")[-2:])[-30:]

        content = ""
        n = len(contents)
        for i, c in enumerate(contents):
            c = c.replace("<br>", "\n").replace("<br/>", "\n")
            content += f"\n\n# {i+1}/{n} ####\n\n{c}"

        send_ntfy(
            url=self.ntfy_url,
            title=f"AnkiMnemonics - '{deckname}'",
            content=content,
        )

    def _load_history(self):
        """
        history structure:
            dict with cid as key and as values:
                a list of subdict:
                    "cardsInfo"  corresponding to the dict of cards info
                        returned by ankiconnect
                    "timestamp" corresponding to the timestamp in seconds
                    "datetime" with the date as string
                    "mnemonics" containing the mnemonics
                    "obsolete" bool that is False if the parsed content of
                        the card has changed
                    "cost" the token cost of the mnemonics
                these subdicts are only added when a card was mnemonize
                    and sent
        """
        whi("Loading history")
        hist_file = Path(MNEMONICS_HIST_PATH)
        if hist_file.exists():
            try:
                self.history = json.load(hist_file.open())
            except Exception as err:
                red(f"Failed to load history json file: '{err}'")
                self.history = {}
        else:
            red("History file not found")
            self.history = {}
        assert isinstance(self.history, dict)
        total_dol = 0
        for cid, hist in self.history.items():
            for i, h in enumerate(hist):
                if "dollar_cost" in h:
                    total_dol += h["dollar_cost"]
                elif "dollar_cost_retroactive" in h:
                    total_dol += h["dollar_cost_retroactive"]
                else:
                    raise ValueError(
                        "Missing dollar_cost (or retroactive) column in history"
                    )

        red(f"Total spending so far: ${total_dol:.2f}")

        return

    def _save_history(self):
        """Save mnemonics history to JSON file.
        
        Saves the history dict to MNEMONICS_HIST_PATH using a temporary file
        to ensure atomic writes. The history contains card info, timestamps,
        mnemonics text and cost data for each card that has been processed.
        """
        hist_p = str(MNEMONICS_HIST_PATH.absolute())
        while hist_p.endswith("/"):
            hist_p = hist_p[:-1]
        hist_pt = hist_p + "_temp"
        with open(str(Path(hist_pt).absolute()), "w") as f:
            json.dump(self.history, f, indent=2)
        assert Path(hist_pt).exists(), "New temporary history file not found"
        Path(hist_p).unlink(missing_ok=True)
        Path(hist_pt).rename(hist_p)
        assert Path(hist_p).exists(), "New history file not found after renaming temp file"
        assert Path(hist_p).absolute() == MNEMONICS_HIST_PATH.absolute()


if __name__ == "__main__":
    try:
        args, kwargs = fire.Fire(lambda *args, **kwargs: [args, kwargs])
        if "help" in kwargs:
            print(help(AnkiMnemonics))
            raise SystemExit()
        AnkiMnemonics(*args, **kwargs)
        sync_anki()
    except Exception as err:
        print(f"Exception: '{err}'")
        if "ntfy_url" in kwargs:
            send_ntfy(
                url=kwargs["ntfy_url"],
                title="AnkiMnemonics - 'error'",
                content=str(err),
            )
        sync_anki()
        raise
