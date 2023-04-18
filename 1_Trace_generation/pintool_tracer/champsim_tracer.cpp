



/*! @file
 *  This is an example of the PIN tool that demonstrates some basic PIN APIs 
 *  and could serve as the starting point for developing your first PIN tool
 */

 /* 8 piece of files for each application
 * scatter: 4 threads
 * gather:  4 threads
 */

// TO DO: scatter reach num but not stop

#include "pin.H"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <unistd.h>

#define NUM_INSTR_DESTINATIONS 2
#define NUM_INSTR_SOURCES 4

//#define NUM_MEM_PHASE_S 20000000
//#define NUM_MEM_PHASE_G 20000000

using namespace std;

typedef struct trace_instr_format {
    unsigned long long int ip;  // instruction pointer (program counter) value

    unsigned char is_branch;    // is this branch
    unsigned char branch_taken; // if so, is this taken

    unsigned char destination_registers[NUM_INSTR_DESTINATIONS]; // output registers
    unsigned char source_registers[NUM_INSTR_SOURCES];           // input registers

    unsigned long long int destination_memory[NUM_INSTR_DESTINATIONS]; // output memory
    unsigned long long int source_memory[NUM_INSTR_SOURCES];           // input memory
} trace_instr_format_t;

/* ================================================================== */
// Global variables 
/* ================================================================== */

UINT64 instrCount = 0;

UINT64 thread_id_glob=0;

UINT64 scatter_gather_flag=0; //scatter:1; gather:2; others:0

UINT64 instrCount_scatter[4]={0}; 
UINT64 instrCount_gather[4]={0}; 

UINT64 min_instrCount_scatter;
UINT64 min_instrCount_gather;


FILE* out_s1; 
FILE* out_s2; 
FILE* out_s3; 
FILE* out_s4; 
FILE* out_g1; 
FILE* out_g2; 
FILE* out_g3; 
FILE* out_g4; 

PIN_LOCK pinLock;

bool output_file_closed = false;
bool tracing_on = false;

trace_instr_format_t curr_instr;

/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool", "o", "champsim.trace", 
        "specify file name for Champsim tracer output");

KNOB<UINT64> KnobSkipInstructions(KNOB_MODE_WRITEONCE, "pintool", "s", "0", 
        "How many instructions to skip before tracing begins");

KNOB<UINT64> KnobTraceInstructions(KNOB_MODE_WRITEONCE, "pintool", "t", "1000000", 
        "How many instructions to trace");
/* ===================================================================== */
// Utilities
/* ===================================================================== */

/*!
 *  Print out help message.
 */
INT32 Usage()
{
    cerr << "This tool creates a register and memory access trace" << endl 
        << "Specify the output trace file with -o" << endl 
        << "Specify the number of instructions to skip before tracing with -s" << endl
        << "Specify the number of instructions to trace with -t" << endl << endl;

    cerr << KNOB_BASE::StringKnobSummary() << endl;

    return -1;
}

/* ===================================================================== */
// Analysis routines
/* ===================================================================== */


void file_open(string fileName){
    string file_name_s1=".s1";
    string file_name_s2=".s2";
    string file_name_s3=".s3";
    string file_name_s4=".s4";
    
    string file_name_g1=".g1";
    string file_name_g2=".g2";
    string file_name_g3=".g3";
    string file_name_g4=".g4";

    file_name_s1=(fileName+file_name_s1);
    file_name_s2=(fileName+file_name_s2);
    file_name_s3=(fileName+file_name_s3);
    file_name_s4=(fileName+file_name_s4);

    file_name_g1=(fileName+file_name_g1);
    file_name_g2=(fileName+file_name_g2);
    file_name_g3=(fileName+file_name_g3);
    file_name_g4=(fileName+file_name_g4);

    out_s1 = fopen(file_name_s1.c_str(), "w");
    out_s2 = fopen(file_name_s2.c_str(), "w");
    out_s3 = fopen(file_name_s3.c_str(), "w");
    out_s4 = fopen(file_name_s4.c_str(), "w");

    out_g1 = fopen(file_name_g1.c_str(), "w");
    out_g2 = fopen(file_name_g2.c_str(), "w");
    out_g3 = fopen(file_name_g3.c_str(), "w");
    out_g4 = fopen(file_name_g4.c_str(), "w");

    return;
}

void file_write(UINT64 phase,UINT64 thread){
    //phase: 1, 2
    //thread: 0,1,2,3
    if (phase == 0) return;

    UINT64 file_case=4*(phase-1)+thread;
    
    switch(file_case){
        case 0:
            if (instrCount_scatter[0] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s1);
            instrCount_scatter[0]++;
            break;
        case 1:
            if (instrCount_scatter[1] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s2);
            instrCount_scatter[1]++;
            break;
        case 2:
            if (instrCount_scatter[2] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s3);
            instrCount_scatter[2]++;
            break;
        case 3:
            if (instrCount_scatter[3] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s4);
            instrCount_scatter[3]++;
            break;
        
        case 4:
            if (instrCount_gather[0] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_g1);
            instrCount_gather[0]++;
            break;
        case 5:
            if (instrCount_gather[1] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_g2);
            instrCount_gather[1]++;
            break;
        case 6:
            if (instrCount_gather[2] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_g3);
            instrCount_gather[2]++;
            break;
        case 7:
            if (instrCount_gather[3] >= KnobTraceInstructions.Value()) break;
            fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_g4);
            instrCount_gather[3]++;
            break;
    }
    min_instrCount_scatter=instrCount_scatter[0];
    min_instrCount_gather=instrCount_gather[0];
    for (int i=1;i<4;i++) if (instrCount_scatter[i]<min_instrCount_scatter) min_instrCount_scatter= instrCount_scatter[i];
    for (int i=1;i<4;i++) if (instrCount_gather[i]<min_instrCount_gather) min_instrCount_gather= instrCount_gather[i];

    return;

   // fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s1);
}

void BeginInstruction(VOID *ip, UINT32 op_code, VOID *opstring)
{
    //instrCount++;


    //if (scatter_gather_flag != 0) instrCount++;
    if (scatter_gather_flag != 0)
    {
        if ((scatter_gather_flag == 1) && (min_instrCount_scatter<KnobTraceInstructions.Value())) 
        {
            tracing_on=true;
            //if (tracing_on==1) cout<<"!!!!!!!!!!!!!!!1`"<<endl;

            //cout<<"tracing_on_0:"<<tracing_on<<endl;
            //instrCount_scatter++;
            // if ((min_instrCount_scatter>0) && (min_instrCount_gather%100000==0)){
            //     //tracing_on=true;
            //     cout<<"tracing_on_1:"<<boolalpha<<tracing_on<<endl;
            //     cout<<"scatter:"<<min_instrCount_scatter<<endl;
            // }
        }
        else if ((scatter_gather_flag == 2) && (min_instrCount_gather<KnobTraceInstructions.Value())) 
        {
            tracing_on=true;
         //   instrCount_gather++;
            // if ((min_instrCount_gather>0) && (min_instrCount_gather%100000==0)){
            // cout<<"tracing_on_2:"<<tracing_on<<endl;
            // cout<<"gather:"<<min_instrCount_gather<<endl;
            // }    
        }
        else{
            tracing_on=false;
        }

        //printf("[%p %u %s ", ip, opcode, (char*)opstring);
        //instrCount=instrCount_scatter+instrCount_gather;

    }
    // if (instrCount%10000==0){
    //     printf("%d", (int)instrCount);
    // }
    else
    {
        tracing_on = false;
    }

    if(!tracing_on) 
        return;

    // reset the current instruction
    curr_instr.ip = (unsigned long long int)ip;

    curr_instr.is_branch = 0;
    curr_instr.branch_taken = 0;

    for(int i=0; i<NUM_INSTR_DESTINATIONS; i++) 
    {
        curr_instr.destination_registers[i] = 0;
        curr_instr.destination_memory[i] = 0;
    }

    for(int i=0; i<NUM_INSTR_SOURCES; i++) 
    {
        curr_instr.source_registers[i] = 0;
        curr_instr.source_memory[i] = 0;
    }
}

THREADID cur_id;

void EndInstruction()
{
    //printf("%d]\n", (int)instrCount);

    //printf("\n");

    if(tracing_on==true)
    {
       // tracing_on = true;

        if((min_instrCount_scatter < KnobTraceInstructions.Value())||(min_instrCount_gather<KnobTraceInstructions.Value()))
        {
            // keep tracing
            // output to file
            //fwrite(&curr_instr, sizeof(trace_instr_format_t), 1, out_s1);
            cur_id=PIN_ThreadId(); 
            file_write(scatter_gather_flag, cur_id); 
            //file_write(scatter_gather_flag,thread_id_glob); 
        }
        else
        {
            tracing_on = false;
            //cout<<"tracing_on_8:"<<tracing_on<<endl;
            // close down the file, we're done tracing
            if(!output_file_closed)
            {
                fclose(out_s1);
                output_file_closed = true;
            }

            exit(0);
        }
    }
}

void BranchOrNot(UINT32 taken)
{
    //printf("[%d] ", taken);

    curr_instr.is_branch = 1;
    if(taken != 0)
    {
        curr_instr.branch_taken = 1;
    }
}

void RegRead(UINT32 i, UINT32 index)
{
    if(!tracing_on) return;

    REG r = (REG)i;

    /*
       if(r == 26)
       {
    // 26 is the IP, which is read and written by branches
    return;
    }
    */

    //cout << r << " " << REG_StringShort((REG)r) << " " ;
    //cout << REG_StringShort((REG)r) << " " ;

    //printf("%d ", (int)r);

    // check to see if this register is already in the list
    int already_found = 0;
    for(int i=0; i<NUM_INSTR_SOURCES; i++)
    {
        if(curr_instr.source_registers[i] == ((unsigned char)r))
        {
            already_found = 1;
            break;
        }
    }
    if(already_found == 0)
    {
        for(int i=0; i<NUM_INSTR_SOURCES; i++)
        {
            if(curr_instr.source_registers[i] == 0)
            {
                curr_instr.source_registers[i] = (unsigned char)r;
                break;
            }
        }
    }
}

void RegWrite(REG i, UINT32 index)
{
    if(!tracing_on) return;

    REG r = (REG)i;

    /*
       if(r == 26)
       {
    // 26 is the IP, which is read and written by branches
    return;
    }
    */

    //cout << "<" << r << " " << REG_StringShort((REG)r) << "> ";
    //cout << "<" << REG_StringShort((REG)r) << "> ";

    //printf("<%d> ", (int)r);

    int already_found = 0;
    for(int i=0; i<NUM_INSTR_DESTINATIONS; i++)
    {
        if(curr_instr.destination_registers[i] == ((unsigned char)r))
        {
            already_found = 1;
            break;
        }
    }
    if(already_found == 0)
    {
        for(int i=0; i<NUM_INSTR_DESTINATIONS; i++)
        {
            if(curr_instr.destination_registers[i] == 0)
            {
                curr_instr.destination_registers[i] = (unsigned char)r;
                break;
            }
        }
    }
    /*
       if(index==0)
       {
       curr_instr.destination_register = (unsigned long long int)r;
       }
       */
}

void MemoryRead(VOID* addr, UINT32 index, UINT32 read_size)
{
    if(!tracing_on) return;

    //printf("0x%llx,%u ", (unsigned long long int)addr, read_size);

    // check to see if this memory read location is already in the list
    int already_found = 0;
    for(int i=0; i<NUM_INSTR_SOURCES; i++)
    {
        if(curr_instr.source_memory[i] == ((unsigned long long int)addr))
        {
            already_found = 1;
            break;
        }
    }
    if(already_found == 0)
    {
        for(int i=0; i<NUM_INSTR_SOURCES; i++)
        {
            if(curr_instr.source_memory[i] == 0)
            {
                curr_instr.source_memory[i] = (unsigned long long int)addr;
                break;
            }
        }
    }
}

void MemoryWrite(VOID* addr, UINT32 index)
{
    if(!tracing_on) return;

    //printf("(0x%llx) ", (unsigned long long int) addr);

    // check to see if this memory write location is already in the list
    int already_found = 0;
    for(int i=0; i<NUM_INSTR_DESTINATIONS; i++)
    {
        if(curr_instr.destination_memory[i] == ((unsigned long long int)addr))
        {
            already_found = 1;
            break;
        }
    }
    if(already_found == 0)
    {
        for(int i=0; i<NUM_INSTR_DESTINATIONS; i++)
        {
            if(curr_instr.destination_memory[i] == 0)
            {
                curr_instr.destination_memory[i] = (unsigned long long int)addr;
                break;
            }
        }
    }
    /*
       if(index==0)
       {
       curr_instr.destination_memory = (long long int)addr;
       }
       */
}

/* ===================================================================== */
// Instrumentation callbacks
/* ===================================================================== */

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    // begin each instruction with this function
    UINT32 opcode = INS_Opcode(ins);
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BeginInstruction, IARG_INST_PTR, IARG_UINT32, opcode, IARG_END);

    // instrument branch instructions
    if(INS_IsBranch(ins))
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)BranchOrNot, IARG_BRANCH_TAKEN, IARG_END);

    // instrument register reads
    UINT32 readRegCount = INS_MaxNumRRegs(ins);
    for(UINT32 i=0; i<readRegCount; i++) 
    {
        UINT32 regNum = INS_RegR(ins, i);

        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RegRead,
                IARG_UINT32, regNum, IARG_UINT32, i,
                IARG_END);
    }

    // instrument register writes
    UINT32 writeRegCount = INS_MaxNumWRegs(ins);
    for(UINT32 i=0; i<writeRegCount; i++) 
    {
        UINT32 regNum = INS_RegW(ins, i);

        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RegWrite,
                IARG_UINT32, regNum, IARG_UINT32, i,
                IARG_END);
    }

    // instrument memory reads and writes
    UINT32 memOperands = INS_MemoryOperandCount(ins);

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++) 
    {
        if (INS_MemoryOperandIsRead(ins, memOp)) 
        {
            UINT32 read_size = INS_MemoryReadSize(ins);

            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MemoryRead,
                    IARG_MEMORYOP_EA, memOp, IARG_UINT32, memOp, IARG_UINT32, read_size,
                    IARG_END);
        }
        if (INS_MemoryOperandIsWritten(ins, memOp)) 
        {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)MemoryWrite,
                    IARG_MEMORYOP_EA, memOp, IARG_UINT32, memOp,
                    IARG_END);
        }
    }

    // finalize each instruction with this function
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)EndInstruction, IARG_END);
}


// This routine is executed each time malloc is called.
VOID BeforeScatter( int size, THREADID threadid )
{
    //instrCount_scatter=0;
    PIN_GetLock(&pinLock, threadid+1);
    thread_id_glob=threadid;
    scatter_gather_flag=1;
    //fprintf(out3, "%ld thread %d entered scatter\n",instrCount, threadid);
    //fflush(out3);
    PIN_ReleaseLock(&pinLock);
}

VOID BeforeGather( int size, THREADID threadid )
{
    //instrCount_gather=0;
    PIN_GetLock(&pinLock, threadid+1);
    thread_id_glob=threadid;
    scatter_gather_flag=2;
    //fprintf(out3, "%ld thread %d entered gather\n",instrCount, threadid);
    //fflush(out3);
    PIN_ReleaseLock(&pinLock);
}

VOID ImageLoad(IMG img, VOID* v)
{
    // Walk through the symbols in the symbol table.
    //
    for (SYM sym = IMG_RegsymHead(img); SYM_Valid(sym); sym = SYM_Next(sym))
    {
        string undFuncName = PIN_UndecorateSymbolName(SYM_Name(sym), UNDECORATION_NAME_ONLY);
 
        //  Find the RtlAllocHeap() function.
        if ((undFuncName == "hint_gather_start"))
        {
            RTN allocRtn = RTN_FindByAddress(IMG_LowAddress(img) + SYM_Value(sym));
 
            if (RTN_Valid(allocRtn))
            {
                // Instrument to print the input argument value and the return value.
                RTN_Open(allocRtn);
 
                RTN_InsertCall(allocRtn, IPOINT_BEFORE, AFUNPTR(BeforeGather),
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_THREAD_ID, IARG_END);
 
                RTN_Close(allocRtn);
            }
        }

        else if (undFuncName == "hint_scatter_start" )
        {
            RTN allocRtn = RTN_FindByAddress(IMG_LowAddress(img) + SYM_Value(sym));
 
            if (RTN_Valid(allocRtn))
            {
                // Instrument to print the input argument value and the return value.
                RTN_Open(allocRtn);
 
                RTN_InsertCall(allocRtn, IPOINT_BEFORE, AFUNPTR(BeforeScatter),
                       IARG_FUNCARG_ENTRYPOINT_VALUE, 0,
                       IARG_THREAD_ID, IARG_END);
 
                RTN_Close(allocRtn);
            }
        }
    }
}
 

VOID ThreadStart(THREADID threadid, CONTEXT *ctxt, INT32 flags, VOID *v)
{
    thread_id_glob=threadid;
    PIN_GetLock(&pinLock, threadid+1);
    //fprintf(out, "thread begin %d\n",threadid);
    //fflush(out);
    PIN_ReleaseLock(&pinLock);
}

// This routine is executed every time a thread is destroyed.
VOID ThreadFini(THREADID threadid, const CONTEXT *ctxt, INT32 code, VOID *v)
{
    PIN_GetLock(&pinLock, threadid+1);
    //fprintf(out, "thread end %d code %d\n",threadid, code);
    //fflush(out);
    PIN_ReleaseLock(&pinLock);
    //scatter_gather_flag=0;
}



/*!
 * Print out analysis results.
 * This function is called when the application exits.
 * @param[in]   code            exit code of the application
 * @param[in]   v               value specified by the tool in the 
 *                              PIN_AddFiniFunction function call
 */
VOID Fini(INT32 code, VOID *v)
{
    // close the file if it hasn't already been closed
    if(!output_file_closed) 
    {
        fclose(out_s1);
        output_file_closed = true;
    }
}

/*!
 * The main procedure of the tool.
 * This function is called when the application image is loaded but not yet started.
 * @param[in]   argc            total number of elements in the argv array
 * @param[in]   argv            array of command line arguments, 
 *                              including pin -t <toolname> -- ...
 */
int main(int argc, char *argv[])
{
    // Initialize PIN library. Print help message if -h(elp) is specified
    // in the command line or the command line is invalid 
    PIN_InitLock(&pinLock);

    if( PIN_Init(argc,argv) ) return Usage();

    PIN_InitSymbols();

    //const char* fileName = KnobOutputFile.Value().c_str();

    //out_s1 = fopen(fileName, "w");
    file_open(KnobOutputFile.ValueString());

    if (!out_s1) 
    {
        cout << "Couldn't open output trace file. Exiting." << endl;
        exit(1);
    }

    // Register function to be called to instrument instructions
    INS_AddInstrumentFunction(Instruction, 0);

    IMG_AddInstrumentFunction(ImageLoad, 0);//pengmiao

    PIN_AddThreadStartFunction(ThreadStart, 0);//pengmiao
    PIN_AddThreadFiniFunction(ThreadFini, 0);

    // Register function to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);

    //cerr <<  "===============================================" << endl;
    //cerr <<  "This application is instrumented by the Champsim Trace Generator" << endl;
    //cerr <<  "Trace saved in " << KnobOutputFile.Value() << endl;
    //cerr <<  "===============================================" << endl;

    // Start the program, never returns
    PIN_StartProgram();

    return 0;
}

/* ===================================================================== */
/* eof */
/* ===================================================================== */
