import os
from typing import Optional, Type, List
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit

try:
    from swcpy import SWCClient, SWCConfig
    from swcpy.swc_client import League, Team 
except ImportError:
    raise ImportError("Biblioteka swpcpy nie jest zainstalowana")

config = SWCConfig(backoff=False)
local_swc_client = SWCClient(config)

class HealthCheckInput(BaseModel):
    pass

class HealthCheckTool(BaseTool):
    name: str = "HealthCheck"
    description: str = (
        "Srawdza czy API działa"
    )
    args_schema: Type[HealthCheckInput] = HealthCheckInput
    return_direct: bool = False

    def _run(
            self, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Urzyj narzędzia do sptawdzenia czy API działa"""
        health_check_response = local_swc_client.get_health_check()
        return health_check_response.txt
    
class LeaguesInput(BaseModel):
    league_name: Optional[str] = Field(
        default=None,
        description="Nazwa ligi. Aby pobrać wszystkie, pozostaw puste lub wpisz None"
    )

class ListLeaguesTool(BaseTool):
    name: str = "ListLeagues"
    description: str = (
        "Pobierz listę lig z APi"
        "Wraz z ligami pobiera listę powiązanych z nim zespołów"
    )
    args_schema: Type[LeaguesInput] = LeaguesInput
    return_direct: bool = False

    def _run(
            self, league_name: Optional[str] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[League]:
        """Użyj tego narzędzia aby pobrać listę rozgrywek"""
        list_leagues_response = local_swc_client.list_leagues(
            league_name = league_name
        )
        return list_leagues_response
    
class TeamsInput(BaseModel):
    team_name: Optional[str] = Field(
        default=None,
        description="Nazwa zespołu. Aby pobrać wszystkie, pozostaw puste lub wpisz None"
    )
    league_id: Optional[int] = Field(
        default=None,
        description="Identyfikator ligi w formacie liczbowym"
        "Pozostaw puste aby uzyskać drużyny ze wszystkich lig"
    )

class ListTeamsTool(BaseTool):
    name: str="ListTeams"
    description: str=(
        "Pobierz listę zespołów. Zwracani są również zawodnicy"
    )
    args_schema: Type[TeamsInput] = TeamsInput
    return_direct: bool=False

    def _run(
            self, team_name: Optional[str] = None,
            league_id: Optional[int] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Team]:
        """Użyj tego narzędzia aby uzyskać listę drużyn"""
        list_teams_response = local_swc_client.list_teams(
            team_name=team_name, league_id=league_id
        )
        return list_teams_response
        
    
class SportWorldCentralToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        """Zwraca listę narzędzi"""
        return [HealthCheckTool(), ListLeaguesTool(), ListTeamsTool()]